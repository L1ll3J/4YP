import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from glob import glob
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageChops
import torchvision.transforms.functional as TF
import torch

#Dataset and Preprocessing
class CustomTestingDataset(Dataset):
    def __init__(self, images_path, masks_path, get_disc=False):

        self.images_path = images_path
        self.masks_path = masks_path
        self.num_samples = len(images_path)
        self.get_disc = get_disc

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (512, 512))
        '''Normalise tensity in range [-1,1]'''
        image = (image - 127.5) / 127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)  # (3,512,512)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask < 128, 2, mask)  # cup
        mask = np.where(mask == 128, 1, mask)  # disc - cup = outer ring
        mask = np.where(mask > 128, 0, mask)  # background
        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)  # (1,512,512)

        return image, mask
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
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        image = cv2.resize(image, (512, 512))
        image = self.transform(image)

        return image
    def __len__(self):
        return self.num_samples

# For Unlabelled Data that is To Be (pre-)Processed
class UnlabelledCustomDatasetTBP(Dataset):
    def __init__(self, images_path):
        self.images_path = images_path
        self.num_samples = len(images_path)
        self.transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = self.transform(image)

        return image

    def __len__(self):
        return self.num_samples


#For Unlabelled Data that has already been (pre-)processed
class UnlabelledCustomDatasetP(Dataset):
    def __init__(self, images_path):
        self.images_path = images_path
        self.num_samples = len(images_path)
        self.transform = transforms.Compose([
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (512, 512))
        '''Normalise tensity in range [-1,1]'''
        image = (image - 127.5) / 127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)  # (3,512,512)

        return image

    def __len__(self):
        return self.num_samples

#Exponential Moving Average - Model Parameters
class EMA:
    def __init__(self, model, decay, delay=0):
        self.model = model
        self.decay = decay
        self.delay = delay
        self.steps = 0  # Track the number of training steps
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters with the model parameters
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        # Increment the step counter
        self.steps += 1

        # Only update the shadow parameters if the delay has passed
        if self.steps > self.delay:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = (self.decay * self.shadow[name] + (1.0 - self.decay) * param.data)
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        # Apply the shadow parameters to the model
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        # Restore original parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


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

def segmentation_score(y_true, y_pred, num_classes):
    # returns confusion matrix (TP, FP, TN, FN) for each class, plus a combined class for class 1+2 (disc)
    if y_true.size() != y_pred.size():
        raise ValueError(f'Check dimensions of y_true {y_true.size()} and y_pred {y_pred.size()}')

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


def average_symmetric_square_difference_multi_class(ori_mask, pred_mask, num_classes=3):
    squared_differences_per_class = np.zeros(num_classes)

    for class_id in range(num_classes):
        # Create binary masks for the current class
        ori_mask_binary = (ori_mask == class_id).astype(int)
        pred_mask_binary = (pred_mask == class_id).astype(int)

        # Calculate squared differences for the current class
        squared_differences = (ori_mask_binary - pred_mask_binary) ** 2
        squared_differences_per_class[class_id] = np.mean(squared_differences)

    # Return the average of the squared differences across all classes
    return np.mean(squared_differences_per_class)

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

    def update_weights(self, image, mask, model, device):
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
