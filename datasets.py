import torch
import torchvision
import torchvision.transforms as transforms
from diffusion.utils import uniform_dequantize


def get_dataset(args):
    BACTH_SIZE = args.batch_size
    NUM_WORKERS = args.workers
    print("Using batch size {}".format(BACTH_SIZE))
    print("Using {} number of workers".format(NUM_WORKERS))
    train_transforms = [
        transforms.Resize(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    test_transforms = [transforms.Resize(32), transforms.ToTensor()]
    if args.uniform_dequantize:
        train_transforms.append(uniform_dequantize)
        test_transforms.append(uniform_dequantize)

    if args.dataset == "mnist":
        dataset = torchvision.datasets.MNIST
    elif args.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10
    elif args.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100

    trainset = dataset(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(train_transforms),
    )
    testset = dataset(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(test_transforms),
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BACTH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BACTH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return trainloader, testloader
