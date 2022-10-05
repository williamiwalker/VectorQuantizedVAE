'''
functions to plot MNIST outputs of vqvae
'''


import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from vqvae_utils import rearrange_mnist, PairedMNISTDataset
import numpy as np
import torch

from torchvision.transforms import ToTensor
from torchvision import datasets, transforms, utils


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
        #images = images
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