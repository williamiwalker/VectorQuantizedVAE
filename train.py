import argparse
from pathlib import Path

import numpy as np
import sys
import json

from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils
from torchvision.transforms import ToTensor


from utils import rearrange_mnist, PairedMNISTDataset
from model import VQVAE, GSSOFT


def save_checkpoint(model, optimizer, step, lossTrack, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": lossTrack}
    checkpoint_path = checkpoint_dir/'learned_model.pth'# / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))


def shift(x):
    return x - 0.5


def train_gssoft(args, saveFolder):
    print('args', args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GSSOFT(args.channels, args.latent_dim, args.num_embeddings, args.embedding_dim)
    model.to(device)

    model_name = "{}_C_{}_N_{}_M_{}_D_{}".format(args.model, args.channels, args.latent_dim,
                                                 args.num_embeddings, args.embedding_dim)

    # checkpoint_dir = Path(model_name)
    print('model_name',model_name)
    checkpoint_dir = Path(saveFolder)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # writer = SummaryWriter(log_dir=Path("runs") / model_name)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.resume is not None:
        print("Resume checkpoint from: {}:".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    ######################################################################################
    # INPUT MNIST DATASET
    ######################################################################################

    # get MNIST data
    # Load MNIST
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
    )

    # test_data = datasets.MNIST(
    #     root='data',
    #     train=False,
    #     transform=ToTensor()
    # )

    # Random seeds
    torch.manual_seed(1)

    # Number of Conditionally independent Factors
    num_factors = 2

    # Sub-Sample original dataset
    train_length = 60000
    test_length = 10000

    # Keep Only some digits
    sub_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    num_digits = len(sub_ids)

    # Rearrange MNIST by grouping num_factors Conditionally independent Observations together
    training_data, train_labels = rearrange_mnist(
        train_data.train_data, train_data.train_labels, num_factors, train_length=train_length, sub_ids=sub_ids)
    training_dataset = PairedMNISTDataset(training_data, train_labels)

    # # Rearrange MNIST by grouping num_factors Conditionally independent Observations together
    # test_data, test_labels = rearrange_mnist(
    #     test_data.test_data, test_data.test_labels, num_factors, train_length=test_length, sub_ids=sub_ids)
    # test_dataset = PairedMNISTDataset(test_data, test_labels)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(shift)
    # ])
    #
    # training_dataset = datasets.CIFAR10("./CIFAR10", train=True, download=True,
    #                                     transform=transform)
    #
    # test_dataset = datasets.CIFAR10("./CIFAR10", train=False, download=True,
    #                                 transform=transform)

    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=True)

    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True,
    #                              num_workers=args.num_workers, pin_memory=True)

    num_epochs = args.num_epochs #args.num_training_steps // len(training_dataloader) + 1
    # start_epoch = global_step // len(training_dataloader) + 1

    lossTrack = np.zeros(num_epochs)

    # print('num epochs', num_epochs, args.num_training_steps, len(training_dataloader))

    #####################################################################################

    # N = 3 * 32 * 32
    N = 2 * 28 * 28

    for epoch in range(num_epochs):
        model.train()
        average_logp = average_KL = average_elbo = average_bpd = average_perplexity = 0
        for i, (images, _) in enumerate(tqdm(training_dataloader), 1):
            images = images.to(device)

            dist, KL, perplexity = model(images)
            targets = (images + 0.5) * 255
            targets = targets.long()
            logp = dist.log_prob(targets).sum((1, 2, 3)).mean()
            loss = (KL - logp) / N
            elbo = (KL - logp) / N
            bpd = elbo / np.log(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # print('get samples', model.getDiscreteLatent(images).shape)

            # if global_step % 212 == 0:
            #     save_checkpoint(model, optimizer, global_step, lossTrack, checkpoint_dir)

            average_logp += (logp.item() - average_logp) / i
            average_KL += (KL.item() - average_KL) / i
            average_elbo += (elbo.item() - average_elbo) / i
            average_bpd += (bpd.item() - average_bpd) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

        # writer.add_scalar("logp/train", average_logp, epoch)
        # writer.add_scalar("kl/train", average_KL, epoch)
        # writer.add_scalar("elbo/train", average_elbo, epoch)
        # writer.add_scalar("bpd/train", average_bpd, epoch)
        # writer.add_scalar("perplexity/train", average_perplexity, epoch)

        lossTrack[epoch] = average_elbo

        # model.eval()
        # average_logp = average_KL = average_elbo = average_bpd = average_perplexity = 0
        # for i, (images, _) in enumerate(test_dataloader, 1):
        #     images = images.to(device)
        #
        #     with torch.no_grad():
        #         dist, KL, perplexity = model(images)
        #
        #     targets = (images + 0.5) * 255
        #     targets = targets.long()
        #     logp = dist.log_prob(targets).sum((1, 2, 3)).mean()
        #     elbo = (KL - logp) / N
        #     bpd = elbo / np.log(2)
        #
        #     average_logp += (logp.item() - average_logp) / i
        #     average_KL += (KL.item() - average_KL) / i
        #     average_elbo += (elbo.item() - average_elbo) / i
        #     average_bpd += (bpd.item() - average_bpd) / i
        #     average_perplexity += (perplexity.item() - average_perplexity) / i
        #
        # writer.add_scalar("logp/test", average_logp, epoch)
        # writer.add_scalar("kl/test", average_KL, epoch)
        # writer.add_scalar("elbo/test", average_elbo, epoch)
        # writer.add_scalar("bpd/test", average_bpd, epoch)
        # writer.add_scalar("perplexity/test", average_perplexity, epoch)
        #
        # samples = torch.argmax(dist.logits, dim=-1)
        # grid = utils.make_grid(samples.float() / 255)
        # writer.add_image("reconstructions", grid, epoch)

        print("epoch:{}, logp:{:.3E}, KL:{:.3E}, elbo:{:.3f}, bpd:{:.3f}, perplexity:{:.3f}"
              .format(epoch, average_logp, average_KL, average_elbo, average_bpd, average_perplexity))

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, global_step, lossTrack, checkpoint_dir)

    save_checkpoint(model, optimizer, global_step, lossTrack, checkpoint_dir)


def train_vqvae(args, saveFolder):
    print('args',args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VQVAE(args.channels, args.latent_dim, args.num_embeddings, args.embedding_dim)
    model.to(device)

    model_name = "{}_C_{}_N_{}_M_{}_D_{}".format(args.model, args.channels, args.latent_dim,
                                                 args.num_embeddings, args.embedding_dim)

    #checkpoint_dir = Path(model_name)
    print('model_name',model_name)
    checkpoint_dir = Path(saveFolder)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print('checkpoint_dir',checkpoint_dir)

    # writer = SummaryWriter(log_dir=Path("runs") / model_name)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.resume is not None:
        print("Resume checkpoint from: {}:".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = checkpoint["step"]
    else:
        global_step = 0

    ######################################################################################
    # INPUT MNIST DATASET
    ######################################################################################

    # get MNIST data
    # Load MNIST
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True,
        )

    # test_data = datasets.MNIST(
    #     root='data',
    #     train=False,
    #     transform=ToTensor()
    #     )

    # Random seeds
    torch.manual_seed(1)

    # Number of Conditionally independent Factors
    num_factors = 2

    # Sub-Sample original dataset
    train_length = 60000
    test_length = 10000

    # Keep Only some digits
    sub_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    num_digits = len(sub_ids)

    # Rearrange MNIST by grouping num_factors Conditionally independent Observations together
    training_data, train_labels = rearrange_mnist(
        train_data.train_data, train_data.train_labels, num_factors, train_length=train_length, sub_ids=sub_ids)
    training_dataset = PairedMNISTDataset(training_data, train_labels)

    # # Rearrange MNIST by grouping num_factors Conditionally independent Observations together
    # test_data, test_labels = rearrange_mnist(
    #     test_data.test_data, test_data.test_labels, num_factors, train_length=test_length, sub_ids=sub_ids)
    # test_dataset = PairedMNISTDataset(test_data, test_labels)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Lambda(shift)
    # ])
    #
    # training_dataset = datasets.CIFAR10("./CIFAR10", train=True, download=True,
    #                                     transform=transform)
    #
    # test_dataset = datasets.CIFAR10("./CIFAR10", train=False, download=True,
    #                                 transform=transform)

    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, pin_memory=True)

    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True,
    #                              num_workers=args.num_workers, pin_memory=True)

    num_epochs = args.num_epochs #args.num_training_steps // len(training_dataloader) + 1
    # start_epoch = global_step // len(training_dataloader) + 1

    lossTrack = np.zeros(num_epochs)

    # print('num epochs', num_epochs, args.num_training_steps, len(training_dataloader))

    #####################################################################################

    # N = 3 * 32 * 32
    # KL = args.latent_dim * 8 * 8 * np.log(args.num_embeddings)
    N = 2 * 28 * 28
    KL = args.latent_dim * np.log(args.num_embeddings)

    for epoch in range(0, num_epochs):
        model.train()
        average_logp = average_vq_loss = average_elbo = average_bpd = average_perplexity = 0
        for i, (images, _) in enumerate(tqdm(training_dataloader), 1):
            images = images.to(device)

            # print('image size', images.shape)

            dist, vq_loss, perplexity = model(images)
            # print('image range',torch.max(images),torch.min(images))
            targets = (images + 0.5) * 255
            targets = targets.long()
            logp = dist.log_prob(targets).sum((1, 2, 3)).mean()
            loss = - logp / N + vq_loss
            elbo = (KL - logp) / N
            bpd = elbo / np.log(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # print('get samples', model.getDiscreteLatent(images).shape)


            average_logp += (logp.item() - average_logp) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_elbo += (elbo.item() - average_elbo) / i
            average_bpd += (bpd.item() - average_bpd) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i

        # writer.add_scalar("logp/train", average_logp, epoch)
        # writer.add_scalar("kl/train", KL, epoch)
        # writer.add_scalar("vqloss/train", average_vq_loss, epoch)
        # writer.add_scalar("elbo/train", average_elbo, epoch)
        # writer.add_scalar("bpd/train", average_bpd, epoch)
        # writer.add_scalar("perplexity/train", average_perplexity, epoch)

        lossTrack[epoch] = average_elbo

        # model.eval()
        # average_logp = average_vq_loss = average_elbo = average_bpd = average_perplexity = 0
        # for i, (images, _) in enumerate(test_dataloader, 1):
        #     images = images.to(device)
        #
        #     with torch.no_grad():
        #         dist, vq_loss, perplexity = model(images)
        #
        #     targets = (images + 0.5) * 255
        #     targets = targets.long()
        #     logp = dist.log_prob(targets).sum((1, 2, 3)).mean()
        #     elbo = (KL - logp) / N
        #     bpd = elbo / np.log(2)
        #
        #     average_logp += (logp.item() - average_logp) / i
        #     average_vq_loss += (vq_loss.item() - average_vq_loss) / i
        #     average_elbo += (elbo.item() - average_elbo) / i
        #     average_bpd += (bpd.item() - average_bpd) / i
        #     average_perplexity += (perplexity.item() - average_perplexity) / i
        #
        # writer.add_scalar("logp/test", average_logp, epoch)
        # writer.add_scalar("kl/test", KL, epoch)
        # writer.add_scalar("vqloss/test", average_vq_loss, epoch)
        # writer.add_scalar("elbo/test", average_elbo, epoch)
        # writer.add_scalar("bpd/test", average_bpd, epoch)
        # writer.add_scalar("perplexity/test", average_perplexity, epoch)
        #
        # samples = torch.argmax(dist.logits, dim=-1)
        # grid = utils.make_grid(samples.float() / 255)
        # writer.add_image("reconstructions", grid, epoch)

        print("epoch:{}, logp:{:.3E}, vq loss:{:.3E}, elbo:{:.3f}, bpd:{:.3f}, perplexity:{:.3f}"
              .format(epoch, average_logp, average_vq_loss, average_elbo, average_bpd, average_perplexity))

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, global_step, lossTrack, checkpoint_dir)

    save_checkpoint(model, optimizer, global_step, lossTrack, checkpoint_dir)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
#     parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume.")
#     parser.add_argument("--model", choices=["VQVAE", "GSSOFT"], help="Select model to train (either VQVAE or GSSOFT)")
#     parser.add_argument("--channels", type=int, default=256, help="Number of channels in conv layers.")
#     parser.add_argument("--latent-dim", type=int, default=8, help="Dimension of categorical latents.")
#     parser.add_argument("--num-embeddings", type=int, default=128, help="Number of codebook embeddings size.")
#     parser.add_argument("--embedding-dim", type=int, default=32, help="Dimension of codebook embeddings.")
#     parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate.")
#     parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
#     parser.add_argument("--num-training-steps", type=int, default=250, help="Number of training steps.")
#     args = parser.parse_args()
#     if args.model == "VQVAE":
#         train_vqvae(args)
#     if args.model == "GSSOFT":
#         train_gssoft(args)






SLURM_ARRAY_TASK_ID = sys.argv[1]
print('SLURM_ARRAY_TASK_ID ',SLURM_ARRAY_TASK_ID)

# ARG_FILE_NAME = 'arguments_test_all_0.json'
ARG_FILE_NAME = 'arguments_VQVAE_3.json'
parent_folder = '/nfs/gatsbystor/williamw/svae/'
# parent_folder = '/home/william/mnt/gatsbystor/implicit_generative/'
ARGUMENT_FILE = parent_folder + 'arg_files/' + ARG_FILE_NAME

with open(ARGUMENT_FILE) as json_file:
    ARGS = json.load(json_file)
    print('PARAMETERS ',ARGS[SLURM_ARRAY_TASK_ID])
    paramDict = ARGS[SLURM_ARRAY_TASK_ID]


OUTPUT_FOLDER = paramDict['MAIN_FOLDER'] + '/' + paramDict['SUB_FOLDER']
saveFolder = parent_folder + OUTPUT_FOLDER + '/'

# just check if cuda is available
cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")

# check if using CUDA
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


args = argparse.Namespace(**paramDict)
if args.model == "VQVAE":
    train_vqvae(args, saveFolder)
if args.model == "GSSOFT":
    train_gssoft(args, saveFolder)