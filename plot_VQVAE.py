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
from scipy.optimize import linear_sum_assignment


# plot_folder = '/nfs/ghome/live/williamw/git/brainscore/RPM_brain/'
plot_folder = '/nfs/ghome/live/williamw/git/implicit_generative/'

# add python path to other plotting utils
sys.path.append(plot_folder)

# from helper_functions.plotting_functions import plotFreeEnergy
from helper_modules.plot_functions import plotFreeEnergy


# just check if cuda is available
cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")


SLURM_ARRAY_TASK_ID = sys.argv[1]
print('SLURM_ARRAY_TASK_ID ',SLURM_ARRAY_TASK_ID)

ARG_FILE_NAME = 'arguments_VQVAE_2.json'
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


training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=True)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True,
                             num_workers=args.num_workers, pin_memory=True)


#####################################################################################

# plotFreeEnergy(saveFolder, 'free_energy', np.zeros(8))


# PLOT RECONSTRUCTIONS



# assets = Path("assets")
# assets.mkdir(exist_ok=True)




def plot_VAE_reconst(saveFolder, name, images, outputs):
    # Feed a single batch through the model

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    images = images.reshape(64,56,28).unsqueeze(1)
    outputs = outputs.reshape(64,56,28).unsqueeze(1)

    image_grid = utils.make_grid(((images + 0.5) * 255).cpu().long())
    image_grid = image_grid.numpy()

    # print('image grid size',image_grid.shape,images.shape)

    ax1.set_title("Original", fontsize="large")
    ax1 = ax1.imshow(np.transpose(image_grid, (1, 2, 0)), interpolation='nearest')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

    output_grid = utils.make_grid(outputs.cpu())
    output_grid = output_grid.numpy()

    # print('output shape',output_grid.shape, outputs.shape)

    ax2.set_title("Reconstructed", fontsize="large")
    ax2 = ax2.imshow(np.transpose(output_grid, (1, 2, 0)), interpolation='nearest')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    fig.tight_layout()
    fig.savefig(saveFolder + '/' + name + '.png')
    # fig.savefig(assets / "reconstructions.png", bbox_inches="tight", pad_inches = 0.5)


def digit_predictions(model, dataloader):

    model.eval()

    num_images = len(dataloader.dataset)
    digitPerdiction = np.zeros(num_images)
    trueLabels = np.zeros(num_images)

    curr_ind = 0
    for i, (images, labels) in enumerate(dataloader, 1):
        images = images.to(device)
        batchSize = images.shape[0]
        # print('image shape',images.shape,batchSize, curr_ind, curr_ind + batchSize, model.getDiscreteLatent(images).flatten().numpy().shape)

        digitPerdiction[curr_ind: curr_ind + batchSize] = model.getDiscreteLatent(images).flatten().numpy()
        trueLabels[curr_ind: curr_ind + batchSize] = labels.numpy()

        curr_ind = curr_ind + batchSize

    return digitPerdiction, trueLabels







def permute_prediction(predictions, labels, used_digits):
    # Find Best Prediction in case the train labels are permuted

    # Number Of used Digits
    num_digits = len(used_digits)

    # Score Used to find Opt. Permutation
    scor_tot = np.eye(num_digits)

    for digit_id in range(num_digits):
        digit_cur = used_digits[digit_id]

        # Prediction digit = digit_cur
        idx = (predictions == digit_cur)

        # How many times 'digit_cur' from prediction is mapped to each digit in label.
        # scor_tot[digit_id, :] = (labels[idx].unsqueeze(dim=0) == used_digits.unsqueeze(dim=1)).sum(1)
        scor_tot[digit_id, :] = (np.expand_dims(labels[idx], axis=0) == np.expand_dims(used_digits, axis=1)).sum(1)


    # Maximise scor_tot using Hungarian Algorithm
    _, perm = linear_sum_assignment(-scor_tot)

    return torch.tensor(perm)


def plot_digit_recognition(saveFolder, name, model, dataloader):

    # acc_tot = torch.zeros(num_exp)

    # Deactivate dropouts
    model.eval()

    # Grasp Test set
    # test_images = test_data.test_data[torch.isin(test_data.test_labels, sub_ids)]
    # test_labels = test_data.test_labels[torch.isin(test_data.test_labels, sub_ids)]

    # Reduce training set
    # reduce_training_set = 20000

    # Convert Test datasets
    # test_tmp = torch.tensor(test_images.clone().detach(), dtype=torch.float32)

    # Use Recognition Network to classify digits
    predictions, labels = digit_predictions(model, dataloader)
    # Get digit labels
    # labels = dataloader.labels.numpy()

    num_images = len(dataloader.dataset)
    sub_ids = np.unique(labels)

    # train_predictions = \
    #     torch.argmax(model.recognition_network.forward(train_images[:reduce_training_set].unsqueeze(dim=1)), dim=1)
    # test_predictions = \
    #     torch.argmax(model.recognition_network.forward(test_tmp.unsqueeze(dim=1)), dim=1)

    # Find best permutation between model clusters and digits identity
    perm_opt = permute_prediction(predictions, labels, sub_ids)

    # Permute Labels
    perm_predictions = perm_opt[predictions]

    print('len', len(perm_predictions), len(labels), num_images)

    # Train / Test performances
    accuracy = sum(abs(perm_predictions - labels) < 0.1) / num_images
    # test_accuracy = sum(abs(test_predictions - test_labels) < 0.01) / len(test_labels)

    # Summary
    results = str(np.round(accuracy.numpy(), 2))
    # test_results = str(np.round(test_accuracy.numpy(), 2))

    print('Training Accuracy = ' + str(accuracy))
    # acc_tot[exp_cur] = accuracy

    # print('Training Accuracy = ' + str(np.round(torch.mean(acc_tot).numpy(), 2))
    #       + ' +/- ' + str(np.round(torch.std(acc_tot).numpy(),2)))
    # print('Testing  Accuracy = ' + str(np.round(torch.mean(test_tot).numpy(), 2))
    #       + ' +/- ' + str(np.round(torch.std(test_tot).numpy(),2)))

# get images and reconstructions for plotting VAE reconstructions
model.eval()
for images, _ in test_dataloader:
    # images = images.to(device)
    with torch.no_grad():
        dist, _, _ = model(images)
        outputs = dist.probs.argmax(dim=-1)
    break
plot_VAE_reconst(saveFolder, 'reconstructions', images, outputs)

plot_digit_recognition(saveFolder, 'digit_recognition', model, test_dataloader)

