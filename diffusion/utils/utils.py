import torch


def fill_tail_dims(
    y: torch.Tensor,
    y_like: torch.Tensor,
):
    """Fill in missing trailing dimensions for y according to y_like."""
    return y[(...,) + (None,) * (y_like.dim() - y.dim())]


def preprocess(
    x: torch.tensor,
    logit_transform: bool = True,
    alpha: float = 0.95,
) -> torch.tensor:
    if logit_transform:
        x = alpha + (1 - 2 * alpha) * x
        x = (x / (1 - x)).log()
    else:
        x = (x - 0.5) * 2
    return x


def postprocess(
    x: torch.tensor,
    logit_transform: bool = True,
    alpha: float = 0.95,
    clamp: bool = True,
) -> torch.tensor:
    if logit_transform:
        x = (x.sigmoid() - alpha) / (1 - 2 * alpha)
    else:
        x = x * 0.5 + 0.5
    return x.clamp(min=0.0, max=1.0) if clamp else x


def uniform_dequantize(
    x: torch.tensor,
    nvals: int = 256,
) -> torch.tensor:
    """[0, 1] -> [0, nvals] -> add uniform noise -> [0, 1]"""
    noise = x.new().resize_as_(x).uniform_()
    x = x * (nvals - 1) + noise
    x = x / nvals
    return x


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
