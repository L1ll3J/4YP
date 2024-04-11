import time
import os
import json
from glob import glob
import copy
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from data_aug.data import train_test_split
from UNET.UNet_model import UNet
from monai.losses import DiceCELoss
from utils import *
import torch
from monai.networks.nets import SwinUNETR
import argparse
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset
import wandb
from torchvision import transforms
from itertools import cycle
import albumentations as A
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose, LoadImage, RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandScaleIntensityd, RandZoomd, Rand2DElasticd, RandGaussianNoised,
    Resize, ToTensord, RandAffined, RandAdjustContrastd
)

import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageChops
import torchvision.transforms.functional as TF
from itertools import combinations
import torch.nn.functional as F



def calculate_discriminator_in_channels(base_c, fmap_combinations):
    """
    Calculate the number of input channels for the Discriminator based on the feature maps from UNet.
    Args:
    - base_c (int): The base number of channels in the UNet (config.base_c).
    - fmap_combinations (list of int): Indices indicating which feature maps to use.

    Returns:
    - int: The total number of input channels for the Discriminator.
    """
    #print(fmap_combinations)
    if fmap_combinations == [9]:
        return base_c

    total_channels = 0
    for index in fmap_combinations:
        if index <= 4:  # pX feature maps
            # For pX, the channel size is determined by the corresponding sX
            channels = base_c * (2 ** (index - 1))
        else:  # sX feature maps
            # For sX, the channel size is base_c * 2^(index - 5), since index starts from 5 for sX
            channels = base_c * (2 ** (index - 5))
        total_channels += channels
    return total_channels

def train_discriminator(model, labeled_loader, scaler, device, config, half, unlabelled_set, true_num_iters ):
    unlabelled_x = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{unlabelled_set}/images/*'))

    if half == 1:
        unlabelled_dataset = unlabelled_x[:len(unlabelled_x) // 2]
    else:
        unlabelled_dataset = unlabelled_x[len(unlabelled_x) // 2:]
    unlabelled_dataset = UnlabelledCustomDatasetTBP(unlabelled_dataset)

    unlabelled_loader = DataLoader(dataset=unlabelled_dataset, batch_size=config.batch_size * config.unlabelled_ratio,
                                   shuffle=True, num_workers=1, pin_memory=True)

    class conv_block(nn.Module):
        def __init__(self, in_c, out_c, norm_name):
            super().__init__()

            self.norm_name = norm_name

            self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            # self.norm1 = norm

            self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
            # self.norm2 = norm

            self.relu = nn.ReLU()

        def forward(self, inputs):
            x = self.conv1(inputs)
            # x = self.norm1(x, self.norm_name)
            x = norm(x, self.norm_name)
            x = self.relu(x)

            x = self.conv2(x)
            # x = self.norm2(x, self.norm_name)
            x = norm(x, self.norm_name)
            x = self.relu(x)

            return x

    class encoder_block(nn.Module):
        def __init__(self, in_c, out_c, norm_name):
            super().__init__()

            self.conv = conv_block(in_c, out_c, norm_name)
            self.pool = nn.MaxPool2d((2, 2))

        def forward(self, inputs):
            x = self.conv(inputs)
            p = self.pool(x)

            return x, p


    class Discriminator(nn.Module):
        def __init__(self, base_c, norm_name):
            super(Discriminator, self).__init__()

            # Adjust the first encoder block to match UNet output channels
            self.e1 = encoder_block(3, base_c, norm_name)
            self.e2 = encoder_block(base_c, base_c * 2, norm_name)
            self.e3 = encoder_block(base_c * 2, base_c * 4, norm_name)
            self.e4 = encoder_block(base_c * 4, base_c * 8, norm_name)

            self.dropout_in = nn.Dropout(p=config.input_dropout)
            self.dropout_reg = nn.Dropout(p=config.regularizing_dropout)

            # Additional layers for binary classification
            self.classifier_conv = nn.Conv2d(base_c * 8, base_c * 16, kernel_size=3, padding=1)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier_fc = nn.Linear(base_c * 16, 1)

        def forward(self, x):
            x = self.dropout_in(x)
            _, x = self.e1(x)
            x = self.dropout_reg(x)
            _, x = self.e2(x)
            x = self.dropout_reg(x)
            _, x = self.e3(x)
            x = self.dropout_reg(x)
            _, x = self.e4(x)

            x = self.dropout_reg(x)

            x = F.relu(self.classifier_conv(x))
            x = self.adaptive_pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier_fc(x)
            return x

    # Instantiate the discriminator
    discriminator = Discriminator(base_c=config.base_c, norm_name='instance').to(device)

    # Define the loss function and optimizer for the discriminator
    disc_loss_fn = nn.BCEWithLogitsLoss()
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001,weight_decay=1e-4)

    disc_loss = 0.0

    # For Discriminator Accuracy
    correct_predictions = 0
    total_predictions = 0

    # Determine alpha for discriminator loss
    model.eval()
    discriminator.train()
    # Training loop
    for epoch in range(200):
        total_disc_loss = 0.0
        for (x, y), u in zip(labeled_loader, unlabelled_loader):


            x, y, u = x.to(device), y.to(device), u.to(device)
            # Apply weak augmentation to the labeled and unlabeled data
            x_weak, y_weak = weakaugmentation(x, y)
            u_weak = weakaugmentation(u)

            x_weak, y_weak, u_weak = x_weak.to(device), y_weak.to(device), u_weak.to(device)
            #x_weak, y_weak, u_weak = x.to(device), y.to(device), u.to(device)



            with autocast():
                y_pred = model(x_weak)


            if config.batch_norm_sanitize == True:
                # Save Batch Normalisation layer
                bn_state = copy.deepcopy(
                    [m.running_mean.clone() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])
                bn_state.extend([m.running_var.clone() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])

                with autocast():
                    u_pred = model(u_weak)

                # Restore Batch Normalisation layer
                for i, m in enumerate([m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]):
                    m.running_mean = bn_state[i]
                    m.running_var = bn_state[i + len(bn_state) // 2]

            else:
                with autocast():
                    u_pred = model(u_weak)

            # Preprocess feature maps
            # outputs_labelled = torch.cat(y_pred, dim=1)
            # outputs_unlabelled = torch.cat(u_pred, dim=1)

            # Discriminator labels: 1 for labeled data, 0 for unlabeled data
            true_labels = torch.ones(y_pred.size(0), 1).to(device)
            fake_labels = torch.zeros(u_pred.size(0), 1).to(device)

            # Combine processed feature maps and labels
            discriminator_input = torch.cat([y_pred, u_pred], dim=0)
            discriminator_labels = torch.cat([true_labels, fake_labels], dim=0)

            # Shuffle combined data and labels together
            shuffled_input, shuffled_labels = shuffle_together(discriminator_input, discriminator_labels)

            # Forward pass of the discriminator with shuffled data
            disc_optimizer.zero_grad()
            with autocast():  # Ensure autocast is used for discriminator as well
                disc_pred = discriminator(shuffled_input)
                disc_loss = disc_loss_fn(disc_pred, shuffled_labels)

            print("Training:", disc_pred)
            predicted_labels = (disc_pred > 0.0).float()  # Convert logits to binary predictions (0 or 1)
            correct = (predicted_labels == shuffled_labels).sum().item()
            total = shuffled_labels.size(0)

            correct_predictions += correct
            total_predictions += total

            # Scale down the discriminator loss and backpropagate (retain_graph=True if needed)
            scaler.scale(disc_loss).backward(retain_graph=True)

            # Update discriminator parameters
            scaler.step(disc_optimizer)


            # Update the scale for next iteration
            scaler.update()

            # Record losses

            total_disc_loss += disc_loss.item()


            discriminator_accuracy = correct_predictions / total_predictions

        wandb.log({f"Discriminator Training Loss {unlabelled_set}": total_disc_loss / len(labeled_loader), f"Discriminator Accuracy {unlabelled_set}": discriminator_accuracy, "Epoch": epoch})
    test_discriminator(model, labeled_loader, unlabelled_loader, discriminator, half, config, unlabelled_set)
    #test(model, config, true_num_iters, half, discriminator, unlabelled_set)


def test_discriminator(model, labeled_loader, unlabeled_loader, discriminator, half, config, unlabelled_set):
    model.eval()  # Ensure the model is in evaluation mode
    discriminator.eval()  # Ensure the discriminator is in evaluation mode

    # Lists to store discriminator scores for analysis
    labeled_scores = []
    unlabeled_scores = []
    test_sets = ["REFUGE1train", "REFUGE1test", "DRISHI", "G1020", "ORIGA", "PAPILA"]
    # test_sets.remove(config.training_set)

    model.eval()
    for test_set in test_sets:

        Prediction_logitsPS = []
        #test_y = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{test_set}/masks/both/*'))
        test_x = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{test_set}/images/*'))

        if test_set == config.training_set or test_set == unlabelled_set:
            if half == 1:
                test_x = test_x[len(test_x) // 2:]
                #test_y = test_y[len(test_y) // 2:]
            else:
                test_x = test_x[:len(test_x) // 2]
                #test_y = test_y[:len(test_y) // 2]

        test_dataset = UnlabelledCustomDatasetTBP(test_x)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size * config.unlabelled_ratio, shuffle=True, num_workers=1, pin_memory=True)


        with torch.no_grad():  # No gradients needed for testing
            # First, test with labeled data
            # for (inputs, _) in labeled_loader:
            #     inputs = weakaugmentation(inputs)
            #     inputs = inputs.to(device)  # Ensure inputs are on the correct device
            #     outputs = model(inputs)  # Get model outputs for labeled data
            #     scores = discriminator(outputs)
            #     #labeled_scores.extend(scores)
            #     print("Testing Controlled, labeled:", scores)

            # Next, test with unlabeled data
            for inputs in test_loader:
                inputs = weakaugmentation(inputs)
                inputs = inputs.to(device)  # Ensure inputs are on the correct device
                outputs = model(inputs)  # Get model outputs for unlabeled data
                scores = discriminator(outputs)  # Discriminator scores
                #unlabeled_scores.extend(scores)
                print(f"Testing Controlled - {test_set} :", scores)
                avg = torch.mean(torch.sigmoid(scores)).item()
                Prediction_logitsPS.append(avg)

            Average_Prediction = sum(Prediction_logitsPS)/ len(Prediction_logitsPS)
            wandb.log({
                f"Unlabelled = {unlabelled_set}, Average Discriminator Prediction on {test_set}": Average_Prediction
            })

    # Analysis/Comparison of scores
    # This part is up to you, depending on what specific metrics or comparisons you're interested in.
    # For example, you might calculate and print the mean scores for each group, plot distributions, etc.

    #return labeled_scores, unlabeled_scores

def model_pipeline():
    with wandb.init():
        config = wandb.config
        print(config)
        for i in range(2):

            model,  train_loader, valid_loader,  criterion,  optimizer,  scaler, ct_augment = make(config, i)

            train(model,  train_loader, valid_loader,  optimizer,  criterion, scaler, device, config, ct_augment, i)

            sets = ["REFUGE1train", "REFUGE1test", "DRISHI", "G1020", "ORIGA", "PAPILA"]
            true_num_iters = round(config.num_iters / len(train_loader)) * len(train_loader)

            test(model, config, true_num_iters, i, False)
            for set in sets:
                if set == config.training_set:
                    continue
                else:
                    train_discriminator(model, train_loader, scaler, device, config, i, set, true_num_iters)


def make(config, i):
    #Make the Data
    # Loading the dataset and preprocessing
    train_x = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{config.training_set}/images/*'))
    train_y = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{config.training_set}/masks/both/*'))
    valid_x = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/val/image/*"))
    valid_y = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/val/mask/*"))

    if i == 1:
        train_x = train_x[:len(train_x)//2]
        train_y = train_y[:len(train_y) // 2]
    else:
        train_x = train_x[len(train_x)//2 :]
        train_y = train_y[len(train_y) // 2:]


    train_dataset, valid_dataset = TrainingCustomDataset(train_x, train_y), train_test_split(valid_x, valid_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    #UNLABELLED DATALOADERS GO HERE
    # content_root_dir = os.path.normpath('C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Unlabelled')
    # subdirs = [os.path.join(content_root_dir, o) for o in os.listdir(content_root_dir)
    #            if os.path.isdir(os.path.join(content_root_dir, o))]
    #
    # # List of all content images
    # content_images = []
    # for subdir in subdirs:
    #     content_subdir = os.path.normpath(os.path.join(subdir, 'images'))
    #     for f in glob(os.path.join(content_subdir, '*.png')):
    #         content_images.append(f)
    # unlabelled_dataset = UnlabelledCustomDataset(content_images)



    #unlabelled_x = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{config.unlabelled_set}/images/*'))

    # if i == 1:
    #     unlabelled_dataset = unlabelled_x[:len(unlabelled_x)//2]
    # else:
    #     unlabelled_dataset = unlabelled_x[len(unlabelled_x) // 2:]
    # unlabelled_dataset = UnlabelledCustomDatasetTBP(unlabelled_dataset)
    #
    # unlabelled_loader = DataLoader(dataset=unlabelled_dataset, batch_size=config.batch_size * config.unlabelled_ratio, shuffle=True, num_workers=1, pin_memory=True)

    #FOR ADV, WE MUST REDEFINE THE UNET EACH RUN :(
    class conv_block(nn.Module):
        def __init__(self, in_c, out_c, norm_name):
            super().__init__()

            self.norm_name = norm_name

            self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            # self.norm1 = norm

            self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
            # self.norm2 = norm

            self.relu = nn.ReLU()

        def forward(self, inputs):
            x = self.conv1(inputs)
            # x = self.norm1(x, self.norm_name)
            x = norm(x, self.norm_name)
            x = self.relu(x)

            x = self.conv2(x)
            # x = self.norm2(x, self.norm_name)
            x = norm(x, self.norm_name)
            x = self.relu(x)

            return x

    class encoder_block(nn.Module):
        def __init__(self, in_c, out_c, norm_name):
            super().__init__()

            self.conv = conv_block(in_c, out_c, norm_name)
            self.pool = nn.MaxPool2d((2, 2))

        def forward(self, inputs):
            x = self.conv(inputs)
            p = self.pool(x)

            return x, p

    class decoder_block(nn.Module):
        def __init__(self, in_c, out_c, norm_name):
            super().__init__()

            self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
            self.conv = conv_block(out_c + out_c, out_c, norm_name)

        def forward(self, inputs, skip):
            x = self.up(inputs)
            x = torch.cat([x, skip], axis=1)
            x = self.conv(x)
            return x

    class UNetAdv(nn.Module):
        def __init__(self, in_c, out_c, base_c, norm_name):
            super().__init__()

            """ Encoder """
            self.e1 = encoder_block(in_c, base_c, norm_name)
            self.e2 = encoder_block(base_c, base_c * 2, norm_name)
            self.e3 = encoder_block(base_c * 2, base_c * 4, norm_name)
            self.e4 = encoder_block(base_c * 4, base_c * 8, norm_name)

            """ Bottleneck """
            self.b = conv_block(base_c * 8, base_c * 16, norm_name)

            """ Decoder """
            self.d1 = decoder_block(base_c * 16, base_c * 8, norm_name)
            self.d2 = decoder_block(base_c * 8, base_c * 4, norm_name)
            self.d3 = decoder_block(base_c * 4, base_c * 2, norm_name)
            self.d4 = decoder_block(base_c * 2, base_c, norm_name)

            """ Classifier """
            self.outputs = nn.Conv2d(base_c, out_c, kernel_size=1, padding=0)

        def forward(self, inputs):
            """ Encoder """
            s1, p1 = self.e1(inputs)
            s2, p2 = self.e2(p1)
            s3, p3 = self.e3(p2)
            s4, p4 = self.e4(p3)

            """ Bottleneck """
            b = self.b(p4)

            """ Decoder """
            d1 = self.d1(b, s4)
            d2 = self.d2(d1, s3)
            d3 = self.d3(d2, s2)
            d4 = self.d4(d3, s1)

            outputs = self.outputs(d4)
            return outputs
    #------------------------------UNetAdv FULLY DEFINED

    unet = UNetAdv(in_c=3, out_c=3, base_c=config.base_c, norm_name=config.norm_name)
    model = unet.to(device)

    #num_channels = int(calculate_discriminator_in_channels(config.base_c, config.fmap_combinations))

    class Discriminator(nn.Module):
        def __init__(self, base_c, norm_name):
            super(Discriminator, self).__init__()

            # Adjust the first encoder block to match UNet output channels
            self.e1 = encoder_block(3, base_c, norm_name)
            self.e2 = encoder_block(base_c, base_c * 2, norm_name)
            self.e3 = encoder_block(base_c * 2, base_c * 4, norm_name)
            self.e4 = encoder_block(base_c * 4, base_c * 8, norm_name)

            self.dropout_in = nn.Dropout(p=config.input_dropout)
            self.dropout_reg = nn.Dropout(p=config.regularizing_dropout)

            # Additional layers for binary classification
            self.classifier_conv = nn.Conv2d(base_c * 8, base_c * 16, kernel_size=3, padding=1)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier_fc = nn.Linear(base_c * 16, 1)

        def forward(self, x):
            x = self.dropout_in(x)
            _, x = self.e1(x)
            x = self.dropout_reg(x)
            _, x = self.e2(x)
            x = self.dropout_reg(x)
            _, x = self.e3(x)
            x = self.dropout_reg(x)
            _, x = self.e4(x)

            x = self.dropout_reg(x)

            x = F.relu(self.classifier_conv(x))
            x = self.adaptive_pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier_fc(x)
            return x

    # Instantiate the discriminator
    discriminator = Discriminator(base_c=config.base_c, norm_name=config.norm_name).to(device)

    # Define the loss function and optimizer for the discriminator
    disc_loss_fn = nn.BCEWithLogitsLoss()
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001,weight_decay=1e-4)


    #Make the Loss and Optimiser
    # Setting the loss function
    criterion = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    scaler = GradScaler()  # For AMP

    # Setting the optimizer with the model parameters and learning rate
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,  # Learning rate from your config
        momentum=config.momentum,  # This is the beta hyperparameter from the paper
        weight_decay=config.weight_decay,  # Regularization term, if applicable
        nesterov=True if config.Nesterov == 'True' else False  # Nesterov momentum
    )

    ct_augment = CTAugment()

    return model, train_loader, valid_loader, criterion,  optimizer,  scaler, ct_augment


def shuffle_together(inputs, labels):
    combined = list(zip(inputs, labels))
    random.shuffle(combined)
    shuffled_inputs, shuffled_labels = zip(*combined)
    return torch.stack(shuffled_inputs), torch.stack(shuffled_labels)
def preprocess_feature_maps(feature_maps, fmap_combinations):
    """
    Preprocess the feature maps for input to the discriminator.
    Args:
    - feature_maps (list of torch.Tensor): The list of feature maps output by the U-Net model.
    - fmap_combinations (list of int): Indices indicating which feature maps to use.

    Returns:
    - torch.Tensor: The concatenated feature maps ready for the discriminator.
    """
    if len(fmap_combinations) == 1:
        return torch.cat(feature_maps, dim=1)

    upsampled_maps = []
    cropped_maps = []
    high_res_sizes = []
    low_res_sizes = []

    # Determine which feature maps are high-resolution based on fmap_combinations
    high_res_indices = [i for i in fmap_combinations if i > 4]
    low_res_indices = [i for i in fmap_combinations if i <= 4]

    # Upsample low-resolution maps
    for i in range(len(low_res_indices)):
        upsampled_map = upsample_to_match_size(feature_maps[i])
        low_res_sizes.append(upsampled_map.size()[2:])
        upsampled_maps.append(upsampled_map)

    # Collect high-resolution maps and their sizes
    for i in range(len(low_res_indices),len(high_res_indices) + len(low_res_indices)):
        high_res_map = feature_maps[i]  # -5 to adjust for index offset
        high_res_sizes.append(high_res_map.size()[2:])
        cropped_maps.append(high_res_map)

    # Determine the size of the smallest high-resolution feature map
    if high_res_sizes:
        target_size = min(high_res_sizes, key=lambda size: (size[0], size[1]))
    else:
        target_size = min(low_res_sizes, key=lambda size: (size[0], size[1]))

    # Crop all feature maps to the target size
    final_maps = [crop_to_target(fmap, target_size) for fmap in upsampled_maps + cropped_maps]

    # Concatenate the feature maps along the channel dimension
    concatenated_maps = torch.cat(final_maps, dim=1)

    return concatenated_maps


def upsample_to_match_size(low_res_fmap):
    """
    Upsample the low-resolution feature map by a factor of 2.
    """
    new_size = (low_res_fmap.size(2) * 2, low_res_fmap.size(3) * 2)
    return F.interpolate(low_res_fmap, size=new_size, mode='nearest')


def crop_to_target(fmap, target_size):
    """
    Crop the given feature map to the target size.
    """
    _, _, h, w = fmap.size()
    th, tw = target_size
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    return fmap[:, :, start_h:start_h + th, start_w:start_w + tw]


def train_one_epoch(model,  labeled_loader,
                    optimizer,  loss_fn, scaler,
                    device, config, ct_augment, epoch):

    # Initialize loss values
    total_loss = 0.0
    total_sup_loss = 0.0
    sup_loss = 0.0
    disc_loss = 0.0

    #For Discriminator Accuracy
    correct_predictions = 0
    total_predictions = 0

    # Determine alpha for discriminator loss
    if epoch < 40:
        alpha = 0
    elif epoch < 65:
        alpha = (config.final_alpha / 25) * (epoch - 40)
    else:
        alpha = config.final_alpha

    # Training loop
    for x, y in labeled_loader:
        model.train()
        x, y = x.to(device), y.to(device)
        # Apply weak augmentation to the labeled and unlabeled data
        x_weak, y_weak = weakaugmentation(x,y)


        x_weak, y_weak = x_weak.to(device), y_weak.to(device)

        # Forward pass for labeled data
        optimizer.zero_grad()
        with autocast():
            y_pred = model(x_weak)
            supervised_loss = loss_fn(y_pred, y_weak)

        # Compute UNet loss with discriminator loss component
        combined_unet_loss = supervised_loss

        # Scale down the combined UNet loss and backpropagate (do not retain graph this time)
        scaler.scale(combined_unet_loss).backward()

        # Update UNet parameters
        scaler.step(optimizer)

        # Update the scale for next iteration
        scaler.update()

        # Record losses
        total_sup_loss += supervised_loss.item()
        total_loss += combined_unet_loss.item()



    return total_loss / len(labeled_loader), total_sup_loss / len(labeled_loader)

def train(model,  train_loader, valid_loader, optimizer,  loss_fn,  scaler, device, config, ct_augment, half):

    data_save_path = f'C:/Users/Josh/Desktop/4YP/Discriminator_Testing_/WITHmodel/models/{config.training_set}/_half{half}'
    create_dir(data_save_path + 'Checkpoint')
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/bs_{config.batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/bs_{config.batch_size}_final.pth'
    create_file(checkpoint_path_lowloss)
    create_file(checkpoint_path_final)
    eval_loss_fn = f1_valid_score
    best_valid_score = 0

    num_epochs = round(config.num_iters / len(train_loader))

    #I KNOW THIS SHOULDN'T BE HERE BUT IT'S THE ONLY WAY TO HAVE
    # LOCAL VARIABLES IN THE SCHEDULING FUNCTION :( (outdated pytorch version)
    def custom_cosine_annealing(step):
        ratio = 7 / 16
        return config.learning_rate * math.cos((ratio * math.pi * step) / num_epochs)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=0, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=5e-5, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=custom_cosine_annealing)
    ema = EMA(model, decay=0.9, delay=20)
    try:
        for epoch in range(num_epochs):
            train_loss, sup_loss = train_one_epoch(model,  train_loader, optimizer,  loss_fn,  scaler, device, config, ct_augment, epoch)
            wandb.log({"Total Training Loss": train_loss, "Supervised Training Loss": sup_loss, "Iteration": epoch*(len(train_loader))})

            if epoch > 40:
                scheduler.step()
            ema.update()

            if epoch % config.validevery == 0:

                #Save Batch Normalisation layer
                bn_state = copy.deepcopy(
                    [m.running_mean.clone() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])
                bn_state.extend([m.running_var.clone() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])

                #Evaluate Model on Validation Set
                s_bg, s_outer, s_cup, s_disc, valid_score = evaluate(model, valid_loader, eval_loss_fn, device)
                if epoch % 30 == 0 and epoch > 0:
                    test(model, config, epoch*(len(train_loader)), half, False)

                #log results
                wandb.log({
                    "Validation Background F1": s_bg,
                    "Validation Outer Ring F1": s_outer,
                    "Validation Cup F1": s_cup,
                    "Validation Disc F1": s_disc,
                    "Validation Score": valid_score,
                    "Iteration": epoch*(len(train_loader))
                })

                #Restore Batch Normalisation layer
                for i, m in enumerate([m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]):
                    m.running_mean = bn_state[i]
                    m.running_var = bn_state[i + len(bn_state) // 2]

                """ Saving the model """
                if valid_score > best_valid_score:
                    data_str = f"Valid score improved from {best_valid_score:2.8f} to {valid_score:2.8f}. Saving checkpoint: {checkpoint_path_lowloss}"
                    print(data_str)
                    best_valid_score = valid_score
                    wandb.log({"Best Validation Score": best_valid_score, "Iteration": epoch*(len(train_loader))})
                    torch.save(model.state_dict(), checkpoint_path_lowloss)

                if epoch+1 == num_epochs:
                    torch.save(model.state_dict(), checkpoint_path_final)

    except KeyboardInterrupt:
        print("Training interrupted. Finishing current epoch...")
        torch.save(model.state_dict(), checkpoint_path_final)
        print("Progress saved to:  " + data_save_path)



def test(model, config, true_num_iters, half, discriminator, unlabelled_set=""):

    # test_x = choose_test_set(test_data_num)

    data_save_path = f'C:/Users/Josh/Desktop/4YP/Discriminator_Testing_/WITHmodel/models/{config.training_set}/_half{half}'


    test_sets = ["REFUGE1train", "REFUGE1test", "DRISHI", "G1020", "ORIGA", "PAPILA"]
    #test_sets.remove(config.training_set)

    model.eval()
    for test_set in test_sets:

        test_y = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{test_set}/masks/both/*'))
        test_x = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{test_set}/images/*'))

        if test_set == config.training_set:
            if half == 1:
                test_x = test_x[len(test_x) // 2:]
                test_y = test_y[len(test_y) // 2:]
            else:
                test_x = test_x[:len(test_x) // 2]
                test_y = test_y[:len(test_y) // 2]

        test_dataset = CustomTestingDataset(test_x, test_y)
        dataset_size = len(test_x)

        if discriminator == False:
            create_dir(data_save_path + f'/results/{test_set}')
            metrics_score = np.zeros((dataset_size, 4, 5))
            dsc_scores = {'cup': [], 'disc': []}
            vCDR_errors = []
            for i in tqdm(range(dataset_size)):
                with torch.no_grad():
                    '''Prediction'''
                    image = test_dataset[i][0].unsqueeze(0).to(device)  # (1, 3, 512, 512)
                    ori_mask = test_dataset[i][1].squeeze(0).to(device)  # (512, 512)
                    pred_y = model(image).squeeze(0)  # (3, 512, 512)
                    pred_y = torch.softmax(pred_y, dim=0)  # (3, 512, 512)
                    pred_mask = torch.argmax(pred_y, dim=0).type(torch.int64)  # (512, 512)

                    score = segmentation_score(ori_mask, pred_mask, num_classes=3)
                    metrics_score[i] = score
                    pred_mask = pred_mask.cpu().numpy()  # (512, 512)
                    ori_mask = ori_mask.cpu().numpy()

                    #vCDR and DSC scores
                    pred_vCDR = calculate_vCDR(pred_mask)
                    true_vCDR = calculate_vCDR(ori_mask)

                    vCDR_errors.append(abs(true_vCDR - pred_vCDR))
                    cup_dsc = dice_similarity_coefficient((ori_mask == 2).astype(np.float32),
                                                          (pred_mask == 2).astype(np.float32))
                    disc_dsc = dice_similarity_coefficient((ori_mask >= 1).astype(np.float32),
                                                           (pred_mask >= 1).astype(np.float32))

                    dsc_scores['cup'].append(cup_dsc)
                    dsc_scores['disc'].append(disc_dsc)

                    '''Scale value back to image'''
                    pred_mask = np.where(pred_mask == 2, 255, pred_mask)
                    pred_mask = np.where(pred_mask == 1, 128, pred_mask)
                    ori_mask = np.where(ori_mask == 2, 255, ori_mask)
                    ori_mask = np.where(ori_mask == 1, 128, ori_mask)
                    image = image * 127.5 + 127.5

                    ori_mask, pred_mask = mask_parse(ori_mask), mask_parse(pred_mask)
                    line = np.ones((512, 20, 3)) * 255  # white line
                    '''Create image for us to analyse visually '''
                    cat_images = np.concatenate(
                        [image.squeeze().permute(1, 2, 0).cpu().numpy(), line, ori_mask, line, pred_mask], axis=1)
                    if i % 10 == 0:
                        cv2.imwrite(data_save_path + f'/results/{test_set}/{i}.png', cat_images)

            np.save(data_save_path + f'/results/{test_set}/' + f'test_score', metrics_score)

            f1_record = metrics_score[:, :, 1]
            f1_mean = metrics_score.mean(axis=0)
            f1_std = np.std(f1_record, axis=0)

            iou_record = metrics_score[:, :, 0]
            iou_mean = metrics_score.mean(axis=0)
            iou_std = np.std(f1_record, axis=0)

            recall_record = metrics_score[:, :, 2]
            recall_mean = metrics_score.mean(axis=0)
            recall_std = np.std(f1_record, axis=0)

            precison_record = metrics_score[:, :, 3]
            precision_mean = metrics_score.mean(axis=0)
            precision_std = np.std(f1_record, axis=0)

            accuracy_record = metrics_score[:, :, 4]
            accuracy_mean = metrics_score.mean(axis=0)
            accuracy_std = np.std(f1_record, axis=0)
            avg_cup_dsc = np.mean(dsc_scores['cup'])
            avg_disc_dsc = np.mean(dsc_scores['disc'])
            mae_vCDR = np.mean(vCDR_errors)

            wandb.log({
                "Optic Cup avg DSC" : avg_cup_dsc,
                "Optic Disk avg DSC" : avg_disc_dsc,
                "vCDR MAE" : mae_vCDR,
                "Iteration": true_num_iters,
                "Training Set": config.training_set,
                "Testing Set": test_set,
                f"Optic Cup avg DSC_{test_set}": avg_cup_dsc,
                f"Optic Disk avg DSC_{test_set}": avg_disc_dsc,
                f"vCDR MAE_{test_set}": mae_vCDR
                ,
                "Outer Ring F1 score": f1_mean[1, 1],
                "Cup F1 score": f1_mean[2, 1],
                "Disc F1 score": f1_mean[3, 1],
                "Outer Ring recall score": recall_mean[1, 2],
                "Cup recall score": recall_mean[2, 2],
                "Disc recall score": recall_mean[3, 2],
                "Outer Ring precision score": precision_mean[1, 3],
                "Cup precision score": precision_mean[2, 3],
                "Disc precision score": precision_mean[3, 3],
                "Outer Ring IOU score": iou_mean[1, 0],
                "Cup IOU score": iou_mean[2, 0],
                "Disc IOU score": iou_mean[3, 0],
                "Outer Ring accuracy score": accuracy_mean[1, 4],
                "Cup accuracy score": accuracy_mean[2, 4],
                "Disc accuracy score": accuracy_mean[3, 4]
            })

        else:
            discriminator.eval()
            discriminator_predictions = []

            create_dir(data_save_path + f'/results/{test_set}')
            metrics_score = np.zeros((dataset_size, 4, 5))
            dsc_scores = {'cup': [], 'disc': []}
            vCDR_errors = []
            for i in tqdm(range(dataset_size)):
                with torch.no_grad():
                    '''Prediction'''
                    image = test_dataset[i][0].unsqueeze(0).to(device)  # (1, 3, 512, 512)
                    ori_mask = test_dataset[i][1].squeeze(0).to(device)  # (512, 512)
                    pred_y = model(image).squeeze(0)  # (3, 512, 512)
                    pred_y = torch.softmax(pred_y, dim=0)  # (3, 512, 512)
                    pred_mask = torch.argmax(pred_y, dim=0).type(torch.int64)  # (512, 512)

                    disc_pred = discriminator(pred_y.unsqueeze(0))
                    disc_probs = torch.sigmoid(disc_pred)
                    discriminator_predictions.append(disc_probs.item())

                    ################
                    print(disc_pred)
                    print(disc_probs)
                    ###############

                    score = segmentation_score(ori_mask, pred_mask, num_classes=3)
                    metrics_score[i] = score
                    pred_mask = pred_mask.cpu().numpy()  # (512, 512)
                    ori_mask = ori_mask.cpu().numpy()

                    # vCDR and DSC scores
                    pred_vCDR = calculate_vCDR(pred_mask)
                    true_vCDR = calculate_vCDR(ori_mask)

                    vCDR_errors.append(abs(true_vCDR - pred_vCDR))
                    cup_dsc = dice_similarity_coefficient((ori_mask == 2).astype(np.float32),
                                                          (pred_mask == 2).astype(np.float32))
                    disc_dsc = dice_similarity_coefficient((ori_mask >= 1).astype(np.float32),
                                                           (pred_mask >= 1).astype(np.float32))

                    dsc_scores['cup'].append(cup_dsc)
                    dsc_scores['disc'].append(disc_dsc)

                    '''Scale value back to image'''
                    pred_mask = np.where(pred_mask == 2, 255, pred_mask)
                    pred_mask = np.where(pred_mask == 1, 128, pred_mask)
                    ori_mask = np.where(ori_mask == 2, 255, ori_mask)
                    ori_mask = np.where(ori_mask == 1, 128, ori_mask)
                    image = image * 127.5 + 127.5

                    ori_mask, pred_mask = mask_parse(ori_mask), mask_parse(pred_mask)
                    line = np.ones((512, 20, 3)) * 255  # white line
                    '''Create image for us to analyse visually '''
                    cat_images = np.concatenate(
                        [image.squeeze().permute(1, 2, 0).cpu().numpy(), line, ori_mask, line, pred_mask], axis=1)
                    if i % 10 == 0:
                        cv2.imwrite(data_save_path + f'/results/{test_set}/{i}.png', cat_images)

            np.save(data_save_path + f'/results/{test_set}/' + f'test_score', metrics_score)

            f1_record = metrics_score[:, :, 1]
            f1_mean = metrics_score.mean(axis=0)
            f1_std = np.std(f1_record, axis=0)

            iou_record = metrics_score[:, :, 0]
            iou_mean = metrics_score.mean(axis=0)
            iou_std = np.std(f1_record, axis=0)

            recall_record = metrics_score[:, :, 2]
            recall_mean = metrics_score.mean(axis=0)
            recall_std = np.std(f1_record, axis=0)

            precison_record = metrics_score[:, :, 3]
            precision_mean = metrics_score.mean(axis=0)
            precision_std = np.std(f1_record, axis=0)

            accuracy_record = metrics_score[:, :, 4]
            accuracy_mean = metrics_score.mean(axis=0)
            accuracy_std = np.std(f1_record, axis=0)
            avg_cup_dsc = np.mean(dsc_scores['cup'])
            avg_disc_dsc = np.mean(dsc_scores['disc'])
            mae_vCDR = np.mean(vCDR_errors)

            avg_discriminator_prediction = sum(discriminator_predictions) / len(discriminator_predictions)

            wandb.log({
                f"Unlabelled = {unlabelled_set}, Average Discriminator Prediction on {test_set}": avg_discriminator_prediction
            })



if __name__ == "__main__":
    #Script Initialisation
    #Data Initialisation

    total_layers = 8  # Total number of layers that can be used
    all_indices = list(range(1, total_layers+1))
    fmap_combinations = []

    # Generate all non-empty combinations of layer indices
    for r in range(1, total_layers + 1):
        fmap_combinations.extend(combinations(all_indices, r))
    fmap_combinations = [list(comb) for comb in fmap_combinations]
    #print((fmap_combinations ))

    fmap_combinations = [[4,8]]

    seeding(42)
    sweep_config = {
        'method': 'grid',
    }
    parameters_dict = {
        'optimizer': {
            'value': 'adam'
        },
        'norm_name': {
            'value': 'batch'
        },
        'base_c': {
            'value': 12
        },
        'fmap_combinations': {'values': fmap_combinations},
        'momentum': {'value': 0.9},
        'lambda_u': {'value':1},
        'Nesterov': {'value': True},
        'threshold': {'value': 0.95},
        'training_set': {'values': [ "REFUGE1test", "DRISHI", "G1020", "ORIGA", "PAPILA"]},
         #'unlabelled_set': {'values': ["REFUGE1train", "REFUGE1test", "DRISHI", "G1020", "ORIGA", "PAPILA"]},
        #'training_set': {'value': "REFUGE1train"},
        'unlabelled_ratio': {'value': 1},
        #'weight_decay': {'values':[ 0.001, 0.005]},
        'weight_decay': {'value': 0.0005},
        'num_iters': {'value': 10000},
        'batch_size': {'value': 5},
        'learning_rate': {'value': 0.03},
        'validevery': {'value': 3},
        'final_alpha': {'value': 3},
        'batch_norm_sanitize': {'values': [True]},
        'input_dropout': {'values': [0.0]},
        'regularizing_dropout': {'values': [0.0]}
    }

    sweep_config['parameters'] = parameters_dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sweep_id = wandb.sweep(sweep=sweep_config, project="Discriminator_Testing_WITHmodel_instancenorm")
    wandb.agent(sweep_id, function=model_pipeline)
