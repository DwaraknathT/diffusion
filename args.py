import yaml
import argparse

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)


parser = argparse.ArgumentParser()

# Dataset params
parser.add_argument("--dataset", "-d", default="cifar10")
parser.add_argument("--num_classes", type=int)
parser.add_argument("--workers", "-j", default=4, type=int)
parser.add_argument("--batch_size", "-b", default=128, type=int)
parser.add_argument("--subset", default=None, type=int)

# Model params
parser.add_argument("--model", "-m", default="unet", type=str)

# Optimization params
parser.add_argument("--lr", default=1e-4, type=float)

# Training params
parser.add_argument("--train_time_eps", default=1e-5, type=float)
parser.add_argument("--time_steps", default=1000, type=int)  # Time steps for time embs

# VPSDE params
parser.add_argument("--beta_min", default=0.1, type=float)
parser.add_argument("--beta_max", default=20.0, type=float)

# Sampling params
parser.add_argument("--num_samples", default=64, type=int)
parser.add_argument("--sample_time_eps", default=1e-3, type=float)


# Probability flow ODE params
parser.add_argument("--ode_solver_tol", default=1e-4, type=float)


def get_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
