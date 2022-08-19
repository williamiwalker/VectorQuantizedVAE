'''
plot everything for GSSOFT and VQVAE
'''

from model import VQVAE, GSSOFT
import sys, os
import json
import torch
import argparse
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms, utils
from utils import rearrange_mnist, PairedMNISTDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# plot_folder = '/nfs/ghome/live/williamw/git/brainscore/RPM_brain/'
plot_folder = '/nfs/ghome/live/williamw/git/implicit_generative/'

# add python path to other plotting utils
sys.path.append(plot_folder)

# from helper_functions.plotting_functions import plotFreeEnergy
from helper_modules.plot_functions import plotFreeEnergy



SLURM_ARRAY_TASK_ID = sys.argv[1]
print('SLURM_ARRAY_TASK_ID ',SLURM_ARRAY_TASK_ID)

ARG_FILE_NAME = 'arguments_VQVAE_0.json'
parent_folder = '/nfs/gatsbystor/williamw/svae/'
ARGUMENT_FILE = parent_folder + 'arg_files/' + ARG_FILE_NAME

with open(ARGUMENT_FILE) as json_file:
    ARGS = json.load(json_file)
    print('PARAMETERS ',ARGS[SLURM_ARRAY_TASK_ID])
    paramDict = ARGS[SLURM_ARRAY_TASK_ID]


OUTPUT_FOLDER = paramDict['MAIN_FOLDER'] + '/' + paramDict['SUB_FOLDER']
saveFolder = parent_folder + OUTPUT_FOLDER + '/'

# make arguments from dict to namespace
args = argparse.Namespace(**paramDict)

# create model
if args.model == "VQVAE":
    model = VQVAE(args.channels, args.latent_dim, args.num_embeddings, args.embedding_dim)
if args.model == "GSSOFT":
    model = GSSOFT(args.channels, args.latent_dim, args.num_embeddings, args.embedding_dim)

# load learned model
checkpoint_path = saveFolder + 'learned_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint["model"])


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

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
)

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

# Rearrange MNIST by grouping num_factors Conditionally independent Observations together
test_data, test_labels = rearrange_mnist(
    test_data.test_data, test_data.test_labels, num_factors, train_length=test_length, sub_ids=sub_ids)
test_dataset = PairedMNISTDataset(test_data, test_labels)

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

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True,
                             num_workers=args.num_workers, pin_memory=True)


#####################################################################################


# plotFreeEnergy(saveFolder, 'free_energy', np.zeros(8))


# PLOT RECONSTRUCTIONS



# assets = Path("assets")
# assets.mkdir(exist_ok=True)




def plot_VAE_reconst(saveFolder, name, model, test_dataloader):
    # Feed a single batch through the model

    model.eval()
    for images, _ in test_dataloader:
        # images = images.to(device)
        with torch.no_grad():
            dist, _, _ = model(images)
            outputs = dist.probs.argmax(dim=-1)
        break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    images = images.reshape(64,56,28).unsqueeze(1)
    outputs = outputs.reshape(64,56,28).unsqueeze(1)

    image_grid = utils.make_grid(((images + 0.5) * 255).cpu().long())
    image_grid = image_grid.numpy()

    print('image grid size',image_grid.shape,images.shape)

    ax1.set_title("Original", fontsize="large")
    ax1 = ax1.imshow(np.transpose(image_grid, (1, 2, 0)), interpolation='nearest')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

    output_grid = utils.make_grid(outputs.cpu())
    output_grid = output_grid.numpy()

    print('output shape',output_grid.shape, outputs.shape)

    ax2.set_title("Reconstructed", fontsize="large")
    ax2 = ax2.imshow(np.transpose(output_grid, (1, 2, 0)), interpolation='nearest')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    fig.tight_layout()
    fig.savefig(saveFolder + '/' + name + '.png')
    # fig.savefig(assets / "reconstructions.png", bbox_inches="tight", pad_inches = 0.5)

plot_VAE_reconst(saveFolder, 'reconstructions', model, test_dataloader)