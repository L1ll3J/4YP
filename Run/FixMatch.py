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

#Dataset and Preprocessing
class TrainingCustomDataset(Dataset):
    def __init__(self, images_path, masks_path, get_disc=False):

        self.images_path = images_path
        self.masks_path = masks_path
        self.num_samples = len(images_path)
        self.get_disc = get_disc
        self.transform = transforms.Compose([
            transforms.ToTensor()])


    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        '''Image to Tensor'''
        image = cv2.resize(image, (512, 512)) # (3,512,512)
        image = self.transform(image)


        """ Mask to Tensor """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask < 128, 2, mask)     # cup
        mask = np.where(mask == 128, 1, mask)    # disc - cup = outer ring
        mask = np.where(mask > 128, 0, mask)     # background
        mask = mask.astype(np.int64)

        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask).type(torch.uint8)  # (1,512,512)
        remapped_mask = torch.zeros_like(mask, dtype=torch.uint8)
        remapped_mask[mask == 1] = 128
        remapped_mask[mask == 2] = 255

        mask = remapped_mask

        #Use these to convert mask back and forth between Tensor and PIL
        # temp_mask_pil = TF.to_pil_image(mask)
        # temp_mask_Tensor = (TF.to_tensor(temp_mask_pil) * 2).to(torch.int64)


        return image, mask

    def __len__(self):
        return self.num_samples

class UnlabelledCustomDataset(Dataset):
    def __init__(self, images_path):

        self.images_path = images_path
        self.num_samples = len(images_path)
        self.transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        '''Normalise tensity in range [-1,-1]'''
        image = cv2.resize(image, (512, 512))
        image = self.transform(image)

        return image
    def __len__(self):
        return self.num_samples



#Augmentation
#Weak Augmentation
def apply_transforms_to_data1(image, mask):
    # Transformations that apply to both image and mask
    if random.random() > 0.5:
       image, mask = TF.hflip(image), TF.hflip(mask)
    if random.random() > 0.5:
       image, mask = TF.vflip(image), TF.vflip(mask)
    if random.random() > 0.5:
       angle = random.choice([0, 90, 180, 270])
       image, mask = TF.rotate(image, angle), TF.rotate(mask, angle)

    #Transformations that apply only to the image
    image = TF.gaussian_blur(image, kernel_size=(5, 5), sigma=(0.1, 0.5))
    image = TF.adjust_brightness(image, brightness_factor=1 + random.uniform(-0.2, 0.2))
    image = TF.adjust_contrast(image, contrast_factor=1 + random.uniform(-0.2, 0.2))
    image = TF.adjust_saturation(image, saturation_factor=1 + random.uniform(-0.2, 0.2))

    return image, mask

def apply_transforms_to_data2(image):
    # Transformations that apply to both image and mask
    if random.random() > 0.5:
        image = TF.hflip(image)
    if random.random() > 0.5:
        image = TF.vflip(image)
    if random.random() > 0.5:
        angle = random.choice([0, 90, 180, 270])
        image = TF.rotate(image, angle)

    # Transformations that apply only to the image
    image = TF.gaussian_blur(image, kernel_size=(5, 5), sigma=(0.1, 0.5))
    image = TF.adjust_brightness(image, brightness_factor=1 + random.uniform(-0.2, 0.2))
    image = TF.adjust_contrast(image, contrast_factor=1 + random.uniform(-0.2, 0.2))
    image = TF.adjust_saturation(image, saturation_factor=1 + random.uniform(-0.2, 0.2))

    return image


def weakaugmentation(img, msk = None):
    if msk is None:
        transformed_images = []
        for x in img:
            x_pil = TF.to_pil_image(x)  # Convert tensor to PIL Image
            transformed_image_pil = apply_transforms_to_data2(x_pil)
            transformed_image = 2*TF.to_tensor(transformed_image_pil) - 1
            transformed_images.append(transformed_image)
        final_image = torch.stack(transformed_images)

        #TF.to_tensor() normalises in intensity [0,1], but we want it as [-1,1]
        return final_image


    else:
        transformed_images = []
        transformed_masks = []
        for x, y in zip(img, msk):

            x_pil = TF.to_pil_image(x)  # Convert image tensor to PIL Image
            y_pil = TF.to_pil_image(y)  # Convert mask tensor to PIL Image

            transformed_image_pil, transformed_mask_pil = apply_transforms_to_data1(x_pil, y_pil)
            transformed_image = 2*TF.to_tensor(transformed_image_pil) - 1  # Convert back to tensor
            transformed_mask = (TF.to_tensor(transformed_mask_pil) * 2).to(torch.int64)  # Convert back to tensor

            transformed_images.append(transformed_image)
            transformed_masks.append(transformed_mask)

            # # USED FOR CHECKING AUGMENTATIONS
            # save_path = "C:\\Users\\Josh\\Desktop\\4YP\\Weak_Augment_Live"
            # os.makedirs(save_path, exist_ok=True)
            # file_id = str(random.randint(10000000, 99999999))
            # transformed_image_pil.save(os.path.join(save_path, f"{file_id}.png"))
            # transformed_mask_pil .save(os.path.join(save_path, f"{file_id}.bmp"))
            # # --------------------------------------

        final_image = torch.stack(transformed_images)
        final_mask = torch.stack(transformed_masks).to(dtype=torch.int64)
        # TF.to_tensor() normalises in intensity [0,1], but we want it as [-1,1]
        return final_image, final_mask


#STRONG AUGMENTATION - CTAUGMENT
#CTAUGMENT START
transformations = {
    'Autocontrast': (0, 1),
    'Brightness': (0.05, 1.00),
    'Color': (0, 1.00),
    'Contrast': (0.05, 1.00),
    #'Cutout': (0, 0.5),
    'Equalize': (0.05,1),
    'Invert': (0,1),
    'Posterize': (1, 8),
    #'Rescale': (0.5, 1.0),
    'Rotate': (-45, 45),
    'Sharpness': (0, 1),
    #'ShearX': (-0.3, 0.3),
    #'ShearY': (-0.3, 0.3),
    'Smooth': (0.05,1),
    'Solarize': (0.05, 1),
    #'TranslateX': (-0.3, 0.3),
    #'TranslateY': (-0.3, 0.3)
}

transformation_functions = {
    'Autocontrast': lambda img, mag: apply_autoCon(img, mag),
    'Brightness': lambda img, mag: apply_brightness(img, mag),
    'Color': lambda img, mag: ImageEnhance.Color(img).enhance(mag),
    'Contrast': lambda img, mag: ImageEnhance.Contrast(img).enhance(mag),
    #'Cutout': lambda img, mag: apply_cutout(img, mag),
    'Equalize': lambda img, mag: apply_equalize(img, mag),#
    'Invert': lambda img, mag: apply_invert(img, mag),
    'Posterize': lambda img, mag: ImageOps.posterize(img, int(mag)),
    #'Rescale': lambda img, mag: TF.resized_crop(img, 0, 0, int(img.height * mag), int(img.width * mag), img.size),
    'Rotate': lambda img, mag: apply_rotate(img, mag),
    'Sharpness': lambda img, mag: ImageEnhance.Sharpness(img).enhance(mag),
    #'ShearX': lambda img, mag: img.transform(img.size, Image.AFFINE, (1, mag, 0, 0, 1, 0)),
    #'ShearY': lambda img, mag: img.transform(img.size, Image.AFFINE, (1, 0, 0, mag, 1, 0)),
    'Smooth': lambda img, mag: apply_smooth(img, mag),
    'Solarize': lambda img, mag: ImageOps.solarize(img, mag*255),
    #'TranslateX': lambda img, mag: img.transform(img.size, Image.AFFINE, (1, 0, mag * img.width, 0, 1, 0)),
    #'TranslateY': lambda img, mag: img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, mag * img.height))
}

spatial_transformations = {
    'Rotate',
    #'TranslateX',
    #'TranslateY',
    #'ShearX',
    #'ShearY',
    #'Rescale'
}
def apply_brightness(image, magnitude):
    enhancer = ImageEnhance.Brightness(image)
    # The magnitude is assumed to be a value between 0 and 1 for brightness adjustment
    return enhancer.enhance(magnitude)

# Example implementation for Rotate:
def apply_rotate(image, magnitude):
    return image.rotate(magnitude)

def apply_invert(image, magnitude):
    # Invert the image using ImageOps
    inverted_image = ImageOps.invert(image)

    # Blend the inverted image with the original image using the magnitude as the alpha parameter
    return ImageChops.blend(inverted_image, image, magnitude)

def apply_equalize(image, magnitude):
    # Equalize the image histogram
    equalized_image = ImageOps.equalize(image)

    return ImageChops.blend(equalized_image, image, magnitude)

def apply_autoCon(image, magnitude):
    # Apply Autocontrast
    autoCon_image = ImageOps.autocontrast(image)

    return ImageChops.blend(autoCon_image, image, magnitude)
def apply_cutout(image, magnitude):
    # Convert magnitude to the actual pixels
    width, height = image.size
    cutout_size = int(width * magnitude)

    # Calculate the square's boundaries
    x_center = random.randint(0, width)
    y_center = random.randint(0, height)
    x1 = max(0, x_center - cutout_size // 2)
    x2 = min(width, x_center + cutout_size // 2)
    y1 = max(0, y_center - cutout_size // 2)
    y2 = min(height, y_center + cutout_size // 2)

    # Create the cutout patch as a PIL image filled with gray
    cutout_patch = Image.new('RGB', (x2 - x1, y2 - y1), color=(127, 127, 127))

    # Create a mask for the cutout area
    mask = Image.new('L', (x2 - x1, y2 - y1), color=255)

    # Paste the cutout patch onto the original image using the mask
    image.paste(cutout_patch, (x1, y1), mask)

    return image
def apply_smooth(image, magnitude):

    # If S is 1, return the original image
    if magnitude >= 1:
        return image

    # The radius for GaussianBlur is derived from S.
    # As S approaches 0, the radius should increase, producing a smoother image.
    # This formula can be adjusted depending on how strong you want the smoothing effect to be.
    radius = (1 - magnitude) * 5  # Example of scaling the radius, assuming the max radius as 5
    return image.filter(ImageFilter.GaussianBlur(radius))


# Initialize CTAugment
class CTAugment:
    def __init__(self, num_bins=17):
        self.weights = {t: [1.0 / num_bins] * num_bins for t in transformations}
        self.magnitude_bins = {t: np.linspace(start, end, num=num_bins) for t, (start, end) in transformations.items()}
        self.num_bins = num_bins
        self.decay_rate = 0.99

    def sample_transformation(self, applied_transformations):
        transformation_name = random.choice(list(self.weights.keys() - set(applied_transformations)))
        bin_weights = self.weights[transformation_name]
        bin_index = random.choices(range(self.num_bins), weights=bin_weights, k=1)[0]
        magnitude = self.magnitude_bins[transformation_name][bin_index]
        return transformation_name, magnitude, bin_index

    def update_weights(self, image, mask, model):
        model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():  # No need to track gradients
            # Apply two random transformations
            transformed_image, transformed_mask, applied_transformations = self.apply_augmentations(image, mask, num_transformations=2)
            transformed_image = transformed_image.to(device)
            transformed_mask = transformed_mask.to(device)
            # Get model prediction for the transformed image
            prediction_logits = model(transformed_image.unsqueeze(0)).squeeze(0)
            prediction = torch.softmax(prediction_logits, dim=0)
            prediction = torch.argmax(prediction, dim=0).type(torch.int64).unsqueeze(0)

            # Compute the Dice score as match score
            for transformation, bin_index in applied_transformations:
                #print(transformed_mask.shape, prediction.shape)
                dice_score = self.compute_dice_score(transformed_mask, prediction)
                self.update_single_weight(transformation, bin_index, dice_score)

    def compute_dice_score(self, true_mask, pred_mask, smooth=1e-6):
        # Flatten the mask and prediction to compute Dice score
        true_mask_flat = true_mask.view(-1)
        pred_mask_flat = pred_mask.view(-1)
        intersection = (true_mask_flat * pred_mask_flat).sum()
        dice_score = (2. * intersection + smooth) / (true_mask_flat.sum() + pred_mask_flat.sum() + smooth)
        return dice_score.item()
    def update_single_weight(self, transformation_name, bin_index, dice_score):
        # Calculate the match score
          # Get the probability of the correct class
        # Update the weight for the transformation's magnitude bin
        current_weight = self.weights[transformation_name][bin_index]
        updated_weight = self.decay_rate * current_weight + (1 - self.decay_rate) * dice_score
        self.weights[transformation_name][bin_index] = updated_weight
        # Re-normalize the weights
        self.normalize_weights(transformation_name)

    def normalize_weights(self, transformation_name):
        total_weight = sum(self.weights[transformation_name])
        self.weights[transformation_name] = [w / total_weight for w in self.weights[transformation_name]]

    def apply_augmentations(self, image, mask, num_transformations=len(transformation_functions), u=False):
        if u:
            image = (image+1)/2
            remapped_mask = torch.zeros_like(mask, dtype=torch.uint8)
            remapped_mask[mask == 1] = 128
            remapped_mask[mask == 2] = 255
            mask = remapped_mask
        image = TF.to_pil_image(image)
        mask = TF.to_pil_image(mask)
        applied_transformations = []
        for _ in range(num_transformations):
            transformation_name, magnitude, bin_index = self.sample_transformation(applied_transformations)
            image = transformation_functions[transformation_name](image, magnitude)
            if transformation_name in spatial_transformations:

                mask = transformation_functions[transformation_name](mask, magnitude)

            applied_transformations.append((transformation_name, bin_index))


        # # USED FOR CHECKING AUGMENTATIONS
        # save_dir = "C:\\Users\\Josh\\Desktop\\4YP\\Strong_Augment_Live"
        # subfolder = "light" if num_transformations == 2 else "heavy"
        # save_path = os.path.join(save_dir, subfolder)
        # os.makedirs(save_path, exist_ok=True)
        # file_id = str(random.randint(10000000, 99999999))
        # image.save(os.path.join(save_path, f"{file_id}.png"))
        # mask.save(os.path.join(save_path, f"{file_id}.bmp"))
        # # --------------------------------------

        image = 2 * TF.to_tensor(image) - 1
        mask = (TF.to_tensor(mask) * 2).to(torch.int64)
        # temp_mask_np = np.array(mask)
        # unique, counts = np.unique(temp_mask_np, return_counts=True)
        # pixel_counts = dict(zip(unique, counts))
        # print(pixel_counts)

        if num_transformations == 2:
            return image, mask, applied_transformations
        else:
            return image, mask
#CTAUGMENT FINISHED

def model_pipeline():
    with wandb.init():
        config = wandb.config
        print(config)

        model, train_loader, valid_loader, unlabelled_loader, criterion, optimizer, scaler, ct_augment = make(config)

        train(model, train_loader, valid_loader, unlabelled_loader, optimizer, criterion, scaler, device, config, ct_augment)

        test(model, config)

def make(config):
    #Make the Data
    # Loading the dataset and preprocessing
    train_x = sorted(glob("C:/Users/Josh/Desktop/4YP/REFUGE_4YP-main/RefugeReformatted/REFUGE1-Train-400/Images_comb/*"))
    train_y = sorted(glob("C:/Users/Josh/Desktop/4YP/REFUGE_4YP-main/RefugeReformatted/REFUGE1-Train-400/Masks/*"))
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

        for xi, yi in zip(x, y):
            ct_augment.update_weights(xi, yi, model)

    return total_loss / len(labeled_loader), sup_loss / len(labeled_loader), unsup_loss / len(labeled_loader)

def train(model, train_loader, valid_loader, unlabeled_loader, optimizer, loss_fn, scaler, device, config, ct_augment):
    data_save_path = f'C:/Users/Josh/Desktop/4YP/REFUGE_4YP-main/RefugeReformatted/test/bs_{config.batch_size}_fs_{config.base_c}_validevery{config.validevery}/'
    create_dir(data_save_path + 'Checkpoint')
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/bs_{config.batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/bs_{config.batch_size}_final.pth'
    create_file(checkpoint_path_lowloss)
    create_file(checkpoint_path_final)
    eval_loss_fn = f1_valid_score
    best_valid_score = 0
    for epoch in range(config.num_epochs):
        train_loss, sup_loss, unsup_loss = train_one_epoch(model, train_loader, unlabeled_loader, optimizer, loss_fn, scaler, device, config, ct_augment)
        wandb.log({"Total Training Loss": train_loss, "Supervised Training Loss": sup_loss, "Unsupervised Training Loss": unsup_loss, "Iteration": epoch*(len(train_loader)/config.batch_size)})

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
    data_save_path = f'C:/Users/Josh/Desktop/4YP/REFUGE_4YP-main/RefugeReformatted/test/bs_{config.batch_size}_fs_{config.base_c}_validevery{config.validevery}/'
    test_dataset = train_test_split(test_x, test_y)
    dataset_size = len(test_x)
    test_data_num = 1
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/bs_{config.batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/bs_{config.batch_size}_final.pth'
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

            ori_mask, pred_mask = mask_parse(ori_mask), mask_parse(pred_mask)
            line = np.ones((512, 20, 3)) * 255  # white line
            '''Create image for us to analyse visually '''
            cat_images = np.concatenate(
                [image.squeeze().permute(1, 2, 0).cpu().numpy(), line, ori_mask, line, pred_mask], axis=1)
            if i % 10 == 0:
                cv2.imwrite(data_save_path + f'results{test_data_num}/{i}.png', cat_images)

    np.save(data_save_path + f'results{test_data_num}/' + f'test_score_{test_data_num}', metrics_score)

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
        'unlabelled_ratio': {'value': 7},
        'weight_decay': {'values':[0.0005, 0.001]},
        'num_epochs': {'value': 200},
        'batch_size': {'values': [3, 5]},
        'learning_rate': {'value': 0.03},
        'validevery': {'value': 3}
    }

    sweep_config['parameters'] = parameters_dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sweep_id = wandb.sweep(sweep=sweep_config, project="FixMatch")
    wandb.agent(sweep_id, function=model_pipeline)
