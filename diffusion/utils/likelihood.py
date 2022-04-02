import torch
from torchdiffeq import odeint


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


def to_flat(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.reshape((-1,))


def from_flat(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return x.reshape(shape)


def get_loglikelihood(
    args,
    sde,
    score_function,
    data,
    noise_type="gaussian",
    is_train=True,
):
    def drift_fn(model, x, t):
        """The drift function of the reverse-time SDE."""
        drift = sde.drift(x, t)
        diffusion = sde.diffusion(x, t)
        g2 = torch.square(diffusion)

        scores = model(x, t)
        dx_dt = drift - 0.5 * g2 * scores
        return dx_dt

    def div_fn(model, x, t, noise):
        return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

    shape = data.shape
    if noise_type == "gaussian":
        epsilon = torch.randn_like(data)
    elif noise_type == "rademacher":
        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.0
    else:
        raise NotImplementedError(f"Hutchinson type {noise_type} unknown.")

    def ode_func(t, x):
        global nfe_counter
        nfe_counter += 1

        x.requires_grad_(True)
        sample = from_flat(x[: -shape[0]], shape)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = to_flat(drift_fn(score_function, sample, vec_t))
        logp_grad = to_flat(div_fn(score_function, sample, vec_t, epsilon))
        return torch.cat([drift, logp_grad], axis=0)

    init = torch.cat(
        [to_flat(data), torch.zeros((shape[0],), device=data.device)],
        axis=0,
    )

    global nfe_counter
    nfe_counter = 0

    solution = odeint(
        ode_func,
        init,
        torch.tensor([args.sample_time_eps, 1.0], device=data.device),
        atol=args.ode_solver_tol,
        rtol=args.ode_solver_tol,
        method="rk4",
        options=dict(step_size=0.5),
    )
    zp = solution[-1]
    z = from_flat(zp[: -shape[0]], shape)
    delta_logp = from_flat(zp[-shape[0] :], (shape[0],))
    prior_logp = sde.prior_logp(z)

    loglikehood = prior_logp + delta_logp

    return loglikehood, nfe_counter
