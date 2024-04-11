import time
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


def model_pipeline():
    with wandb.init():
        config = wandb.config
        print(config)

        model, train_loader, valid_loader, unlabelled_loader, criterion, optimizer, scaler, ct_augment = make(config)

        train(model, train_loader, valid_loader, unlabelled_loader, optimizer, criterion, scaler, device, config, ct_augment)

        test(model, config, round(config.num_iters / len(train_loader))*len(train_loader))

def make(config):
    #Make the Data
    # Loading the dataset and preprocessing
    train_x = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{config.training_set}/images/*'))
    train_y = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{config.training_set}/masks/both/*'))
    valid_x = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/val/image/*"))
    valid_y = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/val/mask/*"))


    train_dataset, valid_dataset = TrainingCustomDataset(train_x, train_y), train_test_split(valid_x, valid_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    #UNLABELLED DATALOADERS GO HERE
    content_root_dir = os.path.normpath('C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Unlabelled')
    subdirs = [os.path.join(content_root_dir, o) for o in os.listdir(content_root_dir)
               if os.path.isdir(os.path.join(content_root_dir, o))]

    # List of all content images
    content_images = []
    for subdir in subdirs:
        content_subdir = os.path.normpath(os.path.join(subdir, 'images'))
        for f in glob(os.path.join(content_subdir, '*.png')):
            content_images.append(f)
    unlabelled_dataset = UnlabelledCustomDataset(content_images)

    unlabelled_loader = DataLoader(dataset=unlabelled_dataset, batch_size=config.batch_size * config.unlabelled_ratio, shuffle=True, num_workers=1, pin_memory=True)

    unet = UNet(in_c=3, out_c=3, base_c=config.base_c, norm_name=config.norm_name)
    model = unet.to(device)

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

    return model, train_loader, valid_loader, unlabelled_loader, criterion, optimizer, scaler, ct_augment

def train_one_epoch(model, labeled_loader, unlabeled_loader, optimizer, loss_fn, scaler, device, config, ct_augment):
    total_loss = 0.0
    sup_loss = 0.0
    unsup_loss = 0.0

    for (x, y), u in zip(labeled_loader, unlabeled_loader):
        model.train()
        x, y, u = x.to(device), y.to(device), u.to(device)
        # Apply weak augmentation to the labeled and unlabeled data
        x_weak, y_weak = weakaugmentation(x,y)
        u_weak = weakaugmentation(u)

        x_weak, y_weak, u_weak = x_weak.to(device), y_weak.to(device), u_weak.to(device)

        # Forward pass for labeled data
        optimizer.zero_grad()
        with autocast():
            y_pred = model(x_weak)
            supervised_loss = loss_fn(y_pred, y_weak)

        # Forward pass for weakly-augmented unlabeled data to generate pseudo-labels
        with torch.no_grad():
            u_pred_weak = model(u_weak)
            pseudo_probs = torch.softmax(u_pred_weak, dim=1)
            max_probs, pseudo_labels = torch.max(pseudo_probs, dim=1)
            mask = max_probs.ge(config.threshold).float()

        # Apply strong augmentation to the unlabeled data
        augmented_results = [ct_augment.apply_augmentations(img, mask, u=True) for img, mask in zip(u_weak, pseudo_labels)]
        u_strong = torch.stack([result[0] for result in augmented_results]).to(device)
        pseudo_labels_strong = []
        for result in augmented_results:
            label_tensor = result[1]
            # Ensure the tensor has 3 dimensions, add singleton dimension if necessary
            if label_tensor.dim() == 2:
                label_tensor = label_tensor.unsqueeze(0)
            pseudo_labels_strong.append(label_tensor.to(device))

        pseudo_labels_strong = torch.stack(pseudo_labels_strong).to(device)

        # Calculate unsupervised loss using strongly-augmented unlabeled data
        with autocast():
            u_pred_strong = model(u_strong)
            unsupervised_loss = loss_fn(u_pred_strong, pseudo_labels_strong) * mask

        # Combine supervised and unsupervised loss
        loss = supervised_loss + (config.lambda_u * unsupervised_loss).mean()

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        sup_loss += supervised_loss.item()
        unsup_loss += unsupervised_loss.mean().item()
        total_loss += loss.item()

        #Update CTAugment bin weights
        for xi, yi in zip(x, y):
            ct_augment.update_weights(xi, yi, model, device)

    return total_loss / len(labeled_loader), sup_loss / len(labeled_loader), unsup_loss / len(labeled_loader)

def train(model, train_loader, valid_loader, unlabeled_loader, optimizer, loss_fn, scaler, device, config, ct_augment):
    data_save_path = f'C:/Users/Josh/Desktop/4YP/FixMatchResults/sweep1/models/{config.training_set}_wd_{config.weight_decay}'
    create_dir(data_save_path + 'Checkpoint')
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/bs_{config.batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/bs_{config.batch_size}_final.pth'
    create_file(checkpoint_path_lowloss)
    create_file(checkpoint_path_final)
    eval_loss_fn = f1_valid_score
    best_valid_score = 0

    num_epochs = round(config.num_iters / len(train_loader))

    for epoch in range(num_epochs):
        train_loss, sup_loss, unsup_loss = train_one_epoch(model, train_loader, unlabeled_loader, optimizer, loss_fn, scaler, device, config, ct_augment)
        wandb.log({"Total Training Loss": train_loss, "Supervised Training Loss": sup_loss, "Unsupervised Training Loss": unsup_loss, "Iteration": epoch*(len(train_loader))})

        if epoch % config.validevery == 0:

            #Save Batch Normalisation layer
            bn_state = copy.deepcopy(
                [m.running_mean.clone() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])
            bn_state.extend([m.running_var.clone() for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)])

            #Evaluate Model on Validation Set
            s_bg, s_outer, s_cup, s_disc, valid_score = evaluate(model, valid_loader, eval_loss_fn, device)

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


def test(model, config, true_num_iters):

    # test_x = choose_test_set(test_data_num)
    data_save_path = f'C:/Users/Josh/Desktop/4YP/FixMatchResults/sweep1/models/{config.training_set}_wd_{config.weight_decay}'


    test_sets = ["REFUGE1train", "REFUGE1test", "DRISHI", "G1020", "ORIGA", "PAPILA"]
    #test_sets.remove(config.training_set)

    model.eval()
    for test_set in test_sets:
        test_y = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{test_set}/masks/both/*'))
        test_x = sorted(glob(f'C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Labelled/{test_set}/images/*'))
        test_dataset = CustomTestingDataset(test_x, test_y)
        dataset_size = len(test_x)

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



if __name__ == "__main__":
    #Script Initialisation
    #Data Initialisation
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
            'value': 24
        },
        'momentum': {'value': 0.9},
        'lambda_u': {'value':1},
        'Nesterov': {'value': True},
        'threshold': {'value': 0.95},
        'training_set': {'values': ["REFUGE1train", "REFUGE1test", "DRISHI", "G1020", "ORIGA", "PAPILA"]},
        #'training_set': {'value': "REFUGE1train"},
        'unlabelled_ratio': {'value': 7},
        #'weight_decay': {'values':[ 0.001, 0.005]},
        'weight_decay': {'value': 0.0005},
        'num_iters': {'value': 60000},
        'batch_size': {'value': 2},
        'learning_rate': {'value': 0.03},
        'validevery': {'value': 3}
    }

    sweep_config['parameters'] = parameters_dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sweep_id = wandb.sweep(sweep=sweep_config, project="FixMatchBaseLine")
    wandb.agent(sweep_id, function=model_pipeline)
