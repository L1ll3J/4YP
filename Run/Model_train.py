import time
import json
from glob import glob
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

parser = argparse.ArgumentParser(description='Specify Parameters')

parser.add_argument('lr', metavar='lr', type=float, help='Specify learning rate')
parser.add_argument('b_s', metavar='b_s', type=int, help='Specify bach size')
parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
parser.add_argument('model', metavar='model', type=str, choices=['unet', 'swin_unetr', 'utnet'], help='Specify a model')
parser.add_argument('norm_name', metavar='norm_name',  help='Specify a normalisation method')
# parser.add_argument('model_text', metavar='model_text', type=str, help='Describe your mode')
parser.add_argument('--base_c', metavar='--base_c', default = 24,type=int, help='base_channel which is the first output channel from first conv block')
# swin_unetr paras
parser.add_argument('--depth', metavar='--depth', type=str, default = '[2,2,2,2]',  help='num_depths in swin_unetr')
parser.add_argument('--n_h', metavar='--n_h', type=str, default = '[3,6,12,24]',  help='num_heads in swin_unetr')


def train(model, data, optimizer, loss_fn, device, scaler):
    iteration_loss = 0.0
    model.train()

    for x, y in data:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast():
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        iteration_loss += loss.detach().item()
    iteration_loss = iteration_loss/len(data)
    return iteration_loss


def evaluate(model, data, score_fn, device):
    model.eval()
    val_score= 0
    f1_score_record = np.zeros(4)

    with torch.no_grad():         #! double check
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
    args = parser.parse_args()
    lr, batch_size, gpu_index, model_name, norm_name = args.lr, args.b_s, args.gpu_index, args.model, args.norm_name
    base_c = args.base_c
    depths = args.depth
    depths = json.loads(depths)
    depths = tuple(depths)
    num_heads = args.n_h
    num_heads = json.loads(num_heads)
    num_heads = tuple(num_heads)

    unet = UNet(in_c=3, out_c=3, base_c=base_c, norm_name=norm_name)

    data_save_path = f'C:/Users/Josh/Desktop/4YP/REFUGE_4YP-main/RefugeReformatted/test/1600_{model_name}_{norm_name}_lr_{lr}_bs_{batch_size}_fs_{base_c}/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    create_dir(data_save_path + 'Checkpoint')

    #Data Initialisation
    seeding(42)
    train_x = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/train/image/*"))
    train_y = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/train/mask/*"))
    valid_x = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/val/image/*"))
    valid_y = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/val/mask/*"))
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/lr_{lr}_bs_{batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/lr_{lr}_bs_{batch_size}_final.pth'
    create_file(checkpoint_path_lowloss)
    create_file(checkpoint_path_final)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    train_dataset, valid_dataset = train_test_split(train_x, train_y),  train_test_split(valid_x, valid_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = unet
    model = model.to(device)

    iteration = 2000
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()  # For AMP
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    train_loss_fn = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    eval_loss_fn = f1_valid_score

    wandb.login()
    wandb.init(project="4YP", name="REFUGE_rough_MonAIv4_c24", config={
    "learning_rate": lr,
    "batch_size": batch_size,
    "num_iterations": iteration,
    "optimizer": "Adam"  # or any other optimizer you are using
    })
    """ Training the model """
    best_valid_score = 0.0

    # Define how often to validate
    validate_every_n_batches = 100

    for iteration_n in tqdm(range(iteration)):


        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, train_loss_fn, device, scaler)
        wandb.log({"Training Loss": train_loss, "Iteration": iteration_n})

        end_time = time.time()
        iteration_mins, iteration_secs = train_time(start_time, end_time)

        data_str = f'Iteration: {iteration_n+1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.8f}\n'

        if iteration_n % validate_every_n_batches == 0:

            s_bg, s_outer, s_cup, s_disc, valid_score = evaluate(model, valid_loader, eval_loss_fn, device)

            wandb.log({
                "Validation Background F1": s_bg,
                "Validation Outer Ring F1": s_outer,
                "Validation Cup F1": s_cup,
                "Validation Disc F1": s_disc,
                "Validation Score": valid_score,
                "Iteration": iteration_n
            })


            """ Saving the model """
            if valid_score > best_valid_score:
                data_str = f"Valid score improved from {best_valid_score:2.8f} to {valid_score:2.8f}. Saving checkpoint: {checkpoint_path_lowloss}"
                print(data_str)
                best_valid_score = valid_score
                torch.save(model.state_dict(), checkpoint_path_lowloss)

            if iteration_n+1 == iteration:
                torch.save(model.state_dict(), checkpoint_path_final)

            data_str += f'\t Val Score: {valid_score:.8f}\n'
        print(data_str)

