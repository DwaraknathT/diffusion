import time
import torch
import torch.nn as nn
from torchdiffeq import odeint

import numpy as np
from .base import AbstractSDE
from .registry import register
from ..utils.utils import fill_tail_dims


class VESDE(AbstractSDE):
    # Beta max and min values defined in eqn 34
    sigma_min: float
    sigma_max: float
    N: float
    device: torch.device

    def __init__(
        self,
        args,
        device: torch.device,
    ) -> None:
        super().__init__()
        # Starting and ending betas
        self.sigma_min = args.sigma_min
        self.sigma_max = args.sigma_max
        self.N = args.time_steps
        self.device = device

        # Since we run into numerical errors using t = 0 during
        # numberical integration, we use a small eps.
        # For VESDE, this eps is 1e-3 for sampling and 1e-5 for training.
        self.train_time_eps = args.train_time_eps
        self.sample_time_eps = args.sample_time_eps

        # ODE solver params
        self.ode_solver_tol = args.ode_solver_tol

    def drift(
        self,
        data: torch.tensor,
        t: torch.tensor,
    ) -> torch.tensor:
        """Computes the drift of the VESDE at time t.
        Drift = torch.zeros_like(data)

        Args:
            data (torch.tensor): Input data tensor
            t (torch.tensor): Time index tensor

        Returns:
            torch.tensor: Drift of SDE at time t
        """
        return torch.zeros_like(data)

    def diffusion(
        self,
        data: torch.tensor,
        t: torch.tensor,
    ) -> torch.tensor:
        """Computes the diffusion of the VESDE at time t
        ratio = sigma_max / sigma_min
        Diffusion = sigma_min  * (ratio) ** t * sqrt(2log(ratio))

        Args:
            data (torch.tensor): Input data tensor
            t (torch.tensor): Time index tensor

        Returns:
            torch.tensor: Diffusion of SDE at time t
        """
        ratio = torch.tensor(self.sigma_max / self.sigma_min)
        diffusion = self.sigma_min * (ratio ** t)
        diffusion *= torch.sqrt(2 * torch.log(ratio))
        return fill_tail_dims(diffusion, data)

    def std_t(
        self,
        data: torch.tensor,
        t: torch.tensor,
    ) -> torch.tensor:
        return torch.sqrt(self.var_t(data, t))

    def var_t(
        self,
        data: torch.tensor,
        t: torch.tensor,
    ) -> torch.tensor:
        """Variance of the marginal distribution p_t(x) at time t
        It is given by
        ratio = sigma_max / sigma_min
        var(t) = sigma_min**2 (ratio) ** 2t

        Args:
            t (torch.tensor): Time index tesor

        Returns:
            torch.tensor: Variance of the marginal at time t
        """
        ratio = self.sigma_max / self.sigma_min
        var = (self.sigma_min ** 2) * (ratio ** (2 * t))
        return fill_tail_dims(var, data)

    def mean_t(
        self,
        data: torch.tensor,
        t: torch.tensor,
    ) -> torch.tensor:
        """Mean of the marginal distribution p_t(x) at time t
        It is given by
        mean(t) = data

        Args:
            data (torch.tensor): Input data tensor
            t (torch.tensor): Time index tensor

        Returns:
            torch.tensor: Mean of the marginal at time t
        """
        return data

    def marginal_distribution(
        self,
        data: torch.tensor,
        t: torch.tensor,
    ) -> tuple:
        """Returns the parameters of the marginal distribution p_t(x)

        Args:
            t (torch.tensor): Time index tensor

        Returns:
            tuple(torch.tensor): Mean, variance and std dev of marginal
        """
        mean = self.mean_t(data, t)
        var = self.var_t(data, t)
        std = self.std_t(data, t)
        return mean, var, std

    def sample(
        self,
        score_function: nn.Module,
        sample_shape: tuple,
        noise: torch.tensor = None,
    ) -> torch.tensor:
        def ode_function(t, x):
            global nfe_counter
            nfe_counter = nfe_counter + 1

            t = torch.max(t, torch.tensor(self.sample_time_eps))
            vec_t = torch.ones(x.shape[0], device=x.device) * t
            drift = self.drift(x, t)
            diffusion = self.diffusion(x, t)
            g2 = torch.square(diffusion)
            # Get scores
            scores = score_function(x, vec_t)
            dx_dt = drift - 0.5 * g2 * scores
            return dx_dt

        if noise is None:
            noise = torch.randn(size=sample_shape, device=self.device)

        # NFE counter
        global nfe_counter
        nfe_counter = 0

        # solve the ODE
        start = time.time()
        samples = odeint(
            ode_function,
            noise,
            torch.tensor([1.0, self.sample_time_eps], device=self.device),
            atol=self.ode_solver_tol,
            rtol=self.ode_solver_tol,
            method="dopri5",
        )
        ode_solve_time = time.time() - start

        return samples[-1], nfe_counter, ode_solve_time

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.0
        return logps


@register
def vesde(args, device):
    return VESDE(args, device)
