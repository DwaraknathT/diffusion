import time
import torch
import torch.nn as nn
from torchdiffeq import odeint

from diffusion.sde.base import SDE
from diffusion.sde.registry import register


class VPSDE(SDE):
    # Beta max and min values defined in eqn 34
    beta_min: float
    beta_max: float
    N: float
    device: torch.device

    def __init__(self, args, device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        # Starting and ending betas
        self.beta_min = args.beta_min
        self.beta_max = args.beta_max
        self.N = args.time_steps
        self.device = device

        # Since we run into numerical errors using t = 0 during
        # numberical integration, we use a small eps.
        # For VPSDE, this eps is 1e-3 for sampling and 1e-5 for training.
        self.train_time_eps = args.train_time_eps
        self.sample_time_eps = args.sample_time_eps

        # ODE solver params
        self.ode_solver_tol = args.ode_solver_tol

    def drift(self, data: torch.tensor, t: torch.tensor) -> torch.tensor:
        """Computes the drift of the VPSDE at time t.
        Drift = -1/2 * (beta_min + t (beta_max - beta_min)) * x(0)

        Args:
            data (torch.tensor): Input data tensor
            t (torch.tensor): Time index tensor

        Returns:
            torch.tensor: Drift of SDE at time t
        """
        return -0.5 * (self.beta_min + t * (self.beta_max - self.beta_min)) * data

    def diffusion(self, data: torch.tensor, t: torch.tensor) -> torch.tensor:
        """Computes the diffusion of the VPSDE at time t
        Diffusion = sqrt(beta_min + t (beta_max - beta_min))

        Args:
            data (torch.tensor): Input data tensor
            t (torch.tensor): Time index tensor

        Returns:
            torch.tensor: Diffusion of SDE at time t
        """
        return torch.sqrt(self.beta_min + t * (self.beta_max - self.beta_min))

    def std_t(self, t: torch.tensor) -> torch.tensor:
        return torch.sqrt(self.var_t(t))

    def var_t(self, t: torch.tensor) -> torch.tensor:
        """Variance of the marginal distribution p_t(x) at time t
        It is given by
        var(t) = (1 - 1/2 * t^2 * (beta_max - beta_min) - t * beta_min)

        Args:
            t (torch.tensor): Time index tesor

        Returns:
            torch.tensor: Variance of the marginal at time t
        """
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        var = 1.0 - torch.exp(2.0 * log_mean_coeff)
        return var

    def mean_t(self, data: torch.tensor, t: torch.tensor) -> torch.tensor:
        """Mean of the marginal distribution p_t(x) at time t
        It is given by
        mean(t) = exp(-1/4 * t^2 * (beta_max - beta_min) - 1/2 * t * beta_min) * data

        Args:
            data (torch.tensor): Input data tensor
            t (torch.tensor): Time index tensor

        Returns:
            torch.tensor: Mean of the marginal at time t
        """
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * data
        return mean

    def marginal_distribution(self, data: torch.tensor, t: torch.tensor) -> tuple:
        """Returns the parameters of the marginal distribution p_t(x)

        Args:
            t (torch.tensor): Time index tensor

        Returns:
            tuple(torch.tensor): Mean, variance and std dev of marginal
        """
        log_mean_coeff = (
            -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * data
        var = 1.0 - torch.exp(2.0 * log_mean_coeff)
        std = torch.sqrt(var)
        return mean, var, std

    def sample(
        self, score_function: nn.Module, sample_shape: tuple, noise: torch.tensor = None
    ) -> torch.tensor:

        score_function.eval()

        def ode_function(t, x):
            vec_t = torch.ones(x.shape[0], device=x.device) * t
            drift = self.drift(x, t)
            diffusion = self.diffusion(x, t)
            g2 = torch.square(diffusion)
            # Get scores
            scaled_t = vec_t * (self.N - 1)
            scores = score_function(x, scaled_t)
            dx_dt = drift - 0.5 * g2 * scores
            return dx_dt

        if noise is None:
            noise = torch.randn(size=sample_shape, device=self.device)

        # solve the ODE
        start = time.time()
        samples = odeint(
            ode_function,
            noise,
            torch.tensor([1.0, self.sample_time_eps], device=self.device),
            atol=self.ode_solver_tol,
            rtol=self.ode_solver_tol,
            method="scipy_solver",
            options={"solver": "RK45"},
        )
        ode_solve_time = time.time() - start

        return samples[-1], ode_solve_time


@register
def vpsde(args, device):
    return VPSDE(args, device)
