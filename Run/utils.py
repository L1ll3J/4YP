import os
import random
import numpy as np
import torch
import torch.nn as nn
from glob import glob

def seeding(seed):  # seeding the randomness
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_file(file):
    if not os.path.exists(file):
        open(file, "w")
    else:
        print(f"{file} Exists")


def train_time(start_time, end_time):
    elapsed_time = end_time - start_time
    return divmod(elapsed_time, 60)


def segmentation_score(y_true, y_pred, num_classes):
    # returns confusion matrix (TP, FP, TN, FN) for each class, plus a combined class for class 1+2 (disc)
    if y_true.size() != y_pred.size():
        raise DimensionError(f'Check dimensions of y_true {y_true.size()} and y_pred {y_pred.size()}')

    smooth = 0.00001
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    score_matrix = np.zeros((num_classes + 1, 5))

    for i in range(num_classes):
        tp = np.sum(np.logical_and(y_true == i, y_pred == i))
        fp = np.sum(np.logical_and(y_true != i, y_pred == i))
        tn = np.sum(np.logical_and(y_true != i, y_pred != i))
        fn = np.sum(np.logical_and(y_true == i, y_pred != i))
        accuracy = (tp + tn)/(tp+fp+tn+fn+smooth)
        precision = tp/(tp+fp+smooth)
        recall = tp/(tp+fn+smooth)
        f1 = 2*tp/(2*tp+fp+fn+smooth)
        IoU = tp/(tp+fp+fn+smooth)
        score_matrix[i] = np.array([IoU, f1, recall, precision, accuracy])
    # DISC
    tp = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), np.logical_or(y_pred == 1, y_pred == 2)))
    fp = np.sum(np.logical_and(y_true == 0, np.logical_or(y_pred == 1, y_pred == 2)))
    tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    fn = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), y_pred == 0))
    accuracy = (tp + tn) / (tp + fp + tn + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    IoU = tp / (tp + fp + fn + smooth)
    score_matrix[3] = np.array([IoU, f1, recall, precision, accuracy])

    return score_matrix


def f1_valid_score(y_true, y_pred):
    if y_true.size() != y_pred.size():
        raise ValueError(f'Check dimensions of y_true {y_true.size()} and y_pred {y_pred.size()}')

    smooth = 1e-5
    score_matrix = torch.zeros(4, device=y_true.device)
    for i in range(3):
        true_i = y_true == i
        pred_i = y_pred == i
        tp = torch.logical_and(true_i, pred_i).sum()
        fp = torch.logical_and(~true_i, pred_i).sum()
        fn = torch.logical_and(true_i, ~pred_i).sum()
        f1 = 2 * tp / (2 * tp + fp + fn + smooth)
        score_matrix[i] = f1

    true_1_or_2 = torch.logical_or(y_true == 1, y_true == 2)
    pred_1_or_2 = torch.logical_or(y_pred == 1, y_pred == 2)
    tp = torch.logical_and(true_1_or_2, pred_1_or_2).sum()
    fp = torch.logical_and(y_true == 0, pred_1_or_2).sum()
    fn = torch.logical_and(true_1_or_2, y_pred == 0).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    score_matrix[3] = f1

    return score_matrix.cpu().numpy()


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)                # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask


def norm(input: torch.tensor, norm_name: str):
    if norm_name == 'layer':
        normaliza = nn.LayerNorm(list(input.shape)[1:])
    elif norm_name == 'batch':
        normaliza = nn.BatchNorm2d(list(input.shape)[1])
    elif norm_name == 'instance':
        normaliza = nn.InstanceNorm2d(list(input.shape)[1])

    normaliza = normaliza.to(f'cuda:{input.get_device()}')

    output = normaliza(input)

    return output


def get_lr(step, lr):
    if step < 100:
        lr_ = 5e-5
    if step > 100:
        lr_ = lr + lr * np.cos(2 * np.pi * step / 100)

    return lr_


def choose_test_set(test_data_num):
    test_x = 'nothing'
    if test_data_num == 0:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image/*"))
    elif test_data_num == 1:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_with_center_white_circle/*"))
    elif test_data_num == 2:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_with_corner_white_circle/*"))
    elif test_data_num == 3:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_with_edge_white_circle/*"))
    elif test_data_num == 4:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.1/*"))
    elif test_data_num == 5:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.2/*"))
    elif test_data_num == 6:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.3/*"))
    elif test_data_num == 7:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.4/*"))
    elif test_data_num == 8:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_r_1.5/*"))
    elif test_data_num == 9:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.1/*"))
    elif test_data_num == 10:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.2/*"))
    elif test_data_num == 11:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.3/*"))
    elif test_data_num == 12:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.4/*"))
    elif test_data_num == 13:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_g_1.5/*"))
    elif test_data_num == 14:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.1/*"))
    elif test_data_num == 15:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.2/*"))
    elif test_data_num == 16:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.3/*"))
    elif test_data_num == 17:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.4/*"))
    elif test_data_num == 18:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image in change/image_b_1.5/*"))
    elif test_data_num == 19:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image_match/*"))
    elif test_data_num == 20:
        test_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/greenlight/*"))
    return test_x
