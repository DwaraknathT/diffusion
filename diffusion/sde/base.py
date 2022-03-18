import abc

import torch
import torch.nn as nn


class SDE(abc.ABC):
    """Base class for diffusions"""

    @abc.abstractmethod
    def drift(self, t: float) -> None:
        """Computes the drift of the SDE at time t

        Args:
            t (float): Time index
        """
        pass

    @abc.abstractmethod
    def diffusion(self, t: float) -> None:
        """Computes the diffusion of the SDE at time index t

        Args:
            t (float): Time index
        """
        pass

    def sample(self, score_function: nn.Module, shape: tuple) -> tuple:
        # Sample noise from standard normal
        noise = torch.randn(shape).cuda()
        score_function.eval()

        # Let's define the probability flow ODE to solve the reverse SDE
        # Given in eqn 13 in the SB with SDE paper
        """The ODE function defined as
        dx / dt = [f(x, t) - 0.5 * (g(t) ** 2) (score(x, t))]

        Args:
            t (float): Time index
            x (torch.tensor): Input at time t

        Returns:
            torch.tensor: Gradient dx/dt at time t
        """

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to("cuda").type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            scores = score_function(x, vec_t)
            drift = self.drift(x, t) - 0.5 * self.g2(t) * scores
            return to_flattened_numpy(drift)

        # Solve the ODE from t = 1 to t = 0 (but eps in practive)
        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(
            ode_func,
            (1, self.sample_time_eps),
            to_flattened_numpy(noise),
            rtol=self.ode_solver_tol,
            atol=self.ode_solver_tol,
            method="RK45",
        )
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).cuda().type(torch.float32)

        return x, nfe
