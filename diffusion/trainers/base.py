"""Base trainer to train diffusion models 
"""
import time
import torch
import torch.optim as optim
import torch.utils.data as data
from torchvision.utils import save_image

from diffusion.sde import get_sde
from diffusion.models import get_model

from diffusion.utils import AverageMeter
from diffusion.utils import fill_tail_dims
from diffusion.utils import preprocess, postprocess

from diffusion.trainers.registry import register


class BaseTrainer:
    def __init__(
        self,
        args,
        device: torch.device = torch.device("cpu"),
    ) -> None:

        self.args = args
        self.device = device

        self.sde = get_sde(args, device)
        self.score_function = get_model(args, device)
        # Optimizer and schedulers
        self.optimizer = optim.Adam(self.score_function.parameters(), lr=args.lr)
        # Meters
        self.step = 1
        self.eval_loss = AverageMeter()
        self.train_loss = AverageMeter()
        # Time meters
        self.data_time = AverageMeter()
        self.train_time = AverageMeter()

    def train_step(
        self,
        data: torch.tensor,
    ) -> float:

        data = data.to(self.device)
        # Preprocess
        data = preprocess(data)
        self.score_function.zero_grad(set_to_none=True)
        # Get time steps
        time_steps = torch.rand(data.shape[0]).to(self.device)
        time_steps = (
            time_steps * (1.0 - self.args.train_time_eps) + self.args.train_time_eps
        )
        # Get parameters of marginal distribution
        mean_t, var_t, std_t = self.sde.marginal_distribution(data, time_steps)
        var_t, std_t = fill_tail_dims(var_t, mean_t), fill_tail_dims(std_t, mean_t)
        # fill tail dimensions for variance and std dev

        # Sample from marginal distribution
        noise = torch.randn_like(data)
        perturbed_data = mean_t + std_t * noise

        # Get scores to denoise the perturbed data
        scaled_time_steps = time_steps * (self.sde.N - 1)
        scores = self.score_function(perturbed_data, scaled_time_steps)

        # Regression targets
        targets = -(perturbed_data - mean_t) / var_t
        # Calculate l2 loss between targets and scores
        # Sum loss across data dimensions
        weighted_scores = var_t * torch.square(scores - targets)
        loss = torch.mean(torch.sum(weighted_scores, dim=[1, 2, 3]))

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(
        self,
        data: torch.tensor,
    ) -> float:

        data = data.to(self.device)
        # Preprocess
        data = preprocess(data)
        # Get time steps
        time_steps = torch.rand(data.shape[0]).to(self.device)
        time_steps = (
            time_steps * (1.0 - self.args.train_time_eps) + self.args.train_time_eps
        )
        # Get parameters of marginal distribution
        mean_t, var_t, std_t = self.sde.marginal_distribution(data, time_steps)
        # fill tail dimensions for variance and std dev
        var_t, std_t = fill_tail_dims(var_t, mean_t), fill_tail_dims(std_t, mean_t)

        # Sample from marginal distribution
        noise = torch.randn_like(data)
        perturbed_data = mean_t + std_t * noise

        # Get scores to denoise the perturbed data
        scaled_time_steps = time_steps * (self.sde.N - 1)
        scores = self.score_function(perturbed_data, scaled_time_steps)

        # Regression targets
        targets = -(perturbed_data - mean_t) / var_t.clamp_min(1e-5)
        # Calculate l2 loss between targets and scores
        # Sum loss across data dimensions
        weighted_scores = var_t * torch.square(scores - targets)
        loss = torch.mean(torch.sum(weighted_scores, dim=[1, 2, 3]))

        return loss.item()

    def evaluate(
        self,
        testloader: data.DataLoader,
    ) -> None:

        self.score_function.eval()
        with torch.no_grad():
            for idx, (data, _) in enumerate(testloader):
                eval_loss = self.eval_step(data)
                self.eval_loss.update(eval_loss)
                if idx % self.args.log_step:
                    print(
                        "Eval Step: {} "
                        "Eval Loss: {:.4f}".format(idx, self.eval_loss.avg)
                    )
        print("Eval Loss: {:.4f}".format(self.eval_loss.avg))
        print("================================")

    def train(
        self,
        trainloader: data.DataLoader,
        testloader: data.DataLoader,
    ) -> None:

        self.score_function.train()
        iterator = iter(trainloader)
        start_time = time.time()
        for step in range(self.step, self.args.train_steps + 1):
            try:
                data, _ = iterator.next()
            except:
                iterator = iter(trainloader)
                data, _ = iterator.next()
            # Log data loading time
            self.data_time.update(time.time() - start_time)

            # Train step
            train_loss = self.train_step(data)
            # Log train step time and loss
            self.train_time.update(time.time() - start_time)
            self.train_loss.update(train_loss)

            # Evaluate network
            if step % self.args.eval_step == 0:
                self.evaluate(testloader)
                # Put network in train mode again
                self.score_function.train()

            # Save the network and other config
            if step % self.args.save_step == 0:
                pass
            if step % self.args.log_step == 0:
                print(
                    "Train Step: {} Train Loss: {:.4f}".format(
                        step, self.train_loss.avg
                    )
                )
            if step % self.args.sample_step == 0:
                self.sample(step)

    def sample(self, step) -> None:

        print("Sampling from Reverse SDE")
        self.score_function.eval()
        with torch.no_grad():
            sample_shape = (
                self.args.num_samples,
                self.args.num_channels,
                self.args.img_size,
                self.args.img_size,
            )
            sample, nfe = self.sde.sample(self.score_function, sample_shape)

        print("No of function evaluations: ", nfe)
        save_image(postprocess(sample), "./samples/sample_{}.png".format(step))


@register
def base_trainer(args, device):
    return BaseTrainer(args, device)
