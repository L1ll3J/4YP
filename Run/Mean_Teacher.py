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
import wandb
import torch.nn.functional as F

def model_pipeline():
    with wandb.init():
        config = wandb.config
        print(config)

        model, train_loader, valid_loader, criterion, optimizer, scaler = make(config)

        train(model, train_loader, valid_loader, optimizer, criterion, scaler, config)

        test_y = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/test/mask/*"))
        test_x = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/test/image/*"))
        test_dataset = train_test_split(test_x, test_y)

        test(model, config)

def update_teacher_model(teacher_model, student_model, alpha):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.copy_(alpha*teacher_param.data + (1-alpha)*student_param.data)

def consistency_loss(teacher_pred, student_pred, temp=0.5):
    return F.mse_loss(teacher_pred, student_pred)
def make(config):
    #Make the Data
    # Loading the dataset and preprocessing
    train_x = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/train/image/*"))
    train_y = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/train/mask/*"))
    valid_x = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/val/image/*"))
    valid_y = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/val/mask/*"))


    train_dataset, valid_dataset = train_test_split(train_x, train_y), train_test_split(valid_x, valid_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    unet = UNet(in_c=3, out_c=3, base_c=config.base_c, norm_name=config.norm_name)
    model = unet.to(device)

    #Make the Loss and Optimiser
    # Setting the loss function
    criterion = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    scaler = GradScaler()  # For AMP

    # Setting the optimizer with the model parameters and learning rate
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)


    return model, train_loader, valid_loader, criterion, optimizer, scaler


def train_one_epoch(student_model, teacher_model, train_loader, unlabeled_loader, optimizer, loss_fn, scaler, alpha):
    student_model.train()
    teacher_model.train()
    iteration_loss = 0.0

    # Loop over both labeled and unlabeled data
    for (x_labeled, y_labeled), (x_unlabeled, _) in zip(train_loader, unlabeled_loader):
        optimizer.zero_grad()

        # Forward pass of labeled data through the student
        x_labeled, y_labeled = x_labeled.to(device), y_labeled.to(device)
        with autocast():
            y_pred_labeled = student_model(x_labeled)
            supervised_loss = loss_fn(y_pred_labeled, y_labeled)

        # Forward pass of unlabeled data through both models
        x_unlabeled = x_unlabeled.to(device)
        # Apply transformations for consistency (could be noise, augmentations, etc.)
        # ...
        with autocast(), torch.no_grad():
            # Teacher model predictions use the moving average weights and are detached from the graph
            teacher_pred = teacher_model(x_unlabeled)
        with autocast():
            student_pred = student_model(x_unlabeled)
            unsupervised_loss = consistency_loss(teacher_pred, student_pred)

        # Combine losses for backpropagation
        loss = supervised_loss + unsupervised_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        iteration_loss += loss.detach().item()

        # Update teacher model as an EMA of student model
        update_teacher_model(teacher_model, student_model, alpha)

    iteration_loss /= len(train_loader)
    return iteration_loss

def train(model, train_loader, valid_loader, optimizer, loss_fn, scaler, config):
    data_save_path = f'C:/Users/Josh/Desktop/4YP/REFUGE_4YP-main/RefugeReformatted/test/sweep2/1600_unet_{config.norm_name}_lr_{round(config.learning_rate, -int(np.floor(np.log10(abs(config.learning_rate))))):e}_bs_{config.batch_size}_fs_{config.base_c}_validevery{config.validevery}/'
    create_dir(data_save_path + 'Checkpoint')
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/lr_{round(config.learning_rate, -int(np.floor(np.log10(abs(config.learning_rate))))):e}_bs_{config.batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/lr_{round(config.learning_rate, -int(np.floor(np.log10(abs(config.learning_rate))))):e}_bs_{config.batch_size}_final.pth'
    create_file(checkpoint_path_lowloss)
    create_file(checkpoint_path_final)
    eval_loss_fn = f1_valid_score
    best_valid_score = 0
    for epoch in range(config.num_epochs):
        train_loss = trainoneepoch(model, train_loader, optimizer, loss_fn, scaler)
        wandb.log({"Training Loss": train_loss, "Iteration": epoch*107})

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
                "Iteration": epoch * 107
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
                wandb.log({"Best Validation Score": best_valid_score, "Iteration": epoch*107})
                torch.save(model.state_dict(), checkpoint_path_lowloss)

            if epoch+1 == config.num_epochs:
                torch.save(model.state_dict(), checkpoint_path_final)

def dice_similarity_coefficient(y_true, y_pred):
    smooth = 0.00001
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def calculate_vCDR(mask):
    # Assuming the mask is binary with 1 for disc and 2 for cup
    disc_mask = mask >= 1
    cup_mask = mask == 2

    # Calculate vertical diameters
    vertical_diameter_disc = np.sum(disc_mask, axis=0).max()
    vertical_diameter_cup = np.sum(cup_mask, axis=0).max()

    return vertical_diameter_cup / vertical_diameter_disc if vertical_diameter_disc != 0 else 0

def test(model, config):
    test_y = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/test/mask/*"))
    test_x = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/test/image/*"))
    # test_x = choose_test_set(test_data_num)
    data_save_path = f'C:/Users/Josh/Desktop/4YP/REFUGE_4YP-main/RefugeReformatted/test/sweep2/1600_unet_{config.norm_name}_lr_{round(config.learning_rate, -int(np.floor(np.log10(abs(config.learning_rate))))):e}_bs_{config.batch_size}_fs_{config.base_c}_validevery{config.validevery}/'
    test_dataset = train_test_split(test_x, test_y)
    dataset_size = len(test_x)
    test_data_num = 1
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/lr_{round(config.learning_rate, -int(np.floor(np.log10(abs(config.learning_rate))))):e}_bs_{config.batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/lr_{round(config.learning_rate, -int(np.floor(np.log10(abs(config.learning_rate))))):e}_bs_{config.batch_size}_final.pth'
    create_dir(data_save_path + f'results{test_data_num}')

    """ Load the checkpoint """
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path_lowloss, map_location=device))
    model.eval()
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
                cv2.imwrite(data_save_path + f'results{test_data_num}/{i}.png', cat_images)

    np.save(data_save_path + f'results{test_data_num}/' + f'test_score_{test_data_num}', metrics_score)

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
        "Iteration": config.num_epochs * 107
    })

def evaluate(model, data, score_fn, device):
    model.eval()
    val_score= 0
    f1_score_record = np.zeros(4)

    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x).softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
            score = score_fn(y, y_pred)
            val_score += score[1].item()/2 + score[2].item()/2
            f1_score_record += score

    f1_score_record /= len(data)
    val_score /= len(data)
    return f1_score_record[0].item(), f1_score_record[1].item(), f1_score_record[2].item(), f1_score_record[3].item(), val_score


if __name__ == "__main__":
    #Script Initialisation
    #Data Initialisation
    seeding(42)

    wandb.login()
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
        'num_epochs': {'value': 400},
        'batch_size': {'value': 15},
        'learning_rate': {'value': 5e-5},
        'validevery': {'value': 5}
    }
    sweep_config['parameters'] = parameters_dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sweep_id = wandb.sweep(sweep=sweep_config, project="Validation_Frequency3")
    wandb.agent(sweep_id, function=model_pipeline)