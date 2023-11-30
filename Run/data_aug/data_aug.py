import os
import re
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from albumentations import HorizontalFlip, VerticalFlip, Rotate

from monai.transforms import (
    Compose, LoadImage, RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandScaleIntensityd, RandZoomd, Rand2DElasticd, RandGaussianNoised,
    Resize, ToTensord, RandAffined, RandAdjustContrastd
)

def random_rotation(image,angle):
    # Randomly choose among 0, 90, 180, 270 degrees
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path_train_x, path_train_y, path_val_x, path_val_y, path_test_x, path_test_y):
    train_x = sorted(glob(os.path.join(path_train_x, '*.jpg')))
    train_y = sorted(glob(os.path.join(path_train_y, '*.bmp')))
    val_x   = sorted(glob(os.path.join(path_val_x, '*.jpg')))
    val_y   = sorted(glob(os.path.join(path_val_y, '*.bmp')))
    test_x  = sorted(glob(os.path.join(path_test_x, '*.jpg')))
    test_y  = sorted(glob(os.path.join(path_test_y, '*.bmp')))

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    keys = ["image", "mask"]
    transforms = Compose([
        #RandFlipd(keys=keys, prob=0.5, spatial_axis=0),  # Horizontal flip
        #RandFlipd(keys=keys, prob=0.5, spatial_axis=1),  # Vertical flip
        #RandRotate90d(keys=keys, prob=0.5, max_k=3),                         #This shit don't work
        #Rand2DElasticd(keys=keys, prob=0.5, spacing=(5, 5), magnitude_range=(2, 4)),  #Introduces Striping
        # RandAffined(
        #     keys=keys,
        #     mode=('bilinear', 'nearest'),  # 'bilinear' for images, 'nearest' for masks to maintain label integrity
        #     prob=0.5,  # probability of applying the transform
        #     rotate_range=np.pi / 12,  # rotation range in radians; this is +/- 15 degrees
        #     scale_range=(0.1, 0.1),  # scale range for both x and y axis; this is a 10% variation
        #     shear_range=np.pi / 18,  # shear range in radians; this is +/- 10 degrees
        #     translate_range=(10, 10),  # translate range in pixels for both x and y axis
        #     padding_mode='reflection' ), # padding mode; options include 'zeros', 'border', or 'reflection'
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandAdjustContrastd( keys=["image"], prob=0.5, gamma=(0.5, 1.5)),  # Range of gamma values to select from
        # ),
        #RandZoomd(keys=keys, prob=0.5, min_zoom=0.9, max_zoom=1.1),
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.03),
        ToTensord(keys=keys)
    ])

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        z = x.replace('\\', '/')
        name = z.split("/")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y)
        #print(x.shape)

        if augment:
            aug = HorizontalFlip(p=1.0)
            x1, y1 = aug(image=x, mask=y)["image"], aug(image=x, mask=y)["mask"]

            aug = VerticalFlip(p=1.0)
            x2, y2 = aug(image=x, mask=y)["image"], aug(image=x, mask=y)["mask"]

            angle = np.random.choice([0, 90, 180, 270])
            x3, y3 = random_rotation(x, angle), random_rotation(y, angle)

            augmented_images = [x, x1, x2, x3]
            augmented_masks = [y, y1, y2, y3]
            transformed_data = []
            for img, mask in zip(augmented_images, augmented_masks):
                data_dict = {"image": img, "mask": mask}
                augmented = transforms(data_dict)

                transformed_image = np.array(augmented["image"]) if isinstance(augmented["image"], np.ndarray) else \
                augmented["image"].numpy()
                transformed_mask = np.array(augmented["mask"]) if isinstance(augmented["mask"], np.ndarray) else \
                augmented["mask"].numpy()


                transformed_data.append((transformed_image, transformed_mask))

            # Step 4: Collect the transformed images and masks
            X = [item[0] for item in transformed_data]
            Y = [item[1] for item in transformed_data]


        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):

            i = cv2.resize(i, size)
            m = cv2.resize(m, size, interpolation= cv2.INTER_NEAREST)

            tmp_image_name = f"{name}_{index}.jpg"
            tmp_mask_name = f"{name}_{index}.bmp"


            image_path = os.path.normpath(os.path.join(save_path, "image", tmp_image_name))
            mask_path = os.path.normpath(os.path.join(save_path, "mask", tmp_mask_name))




            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    ori_data_path = 'C:/Users/Josh/Desktop/4YP/REFUGE_4YP-main/RefugeReformatted/'
    data_path_train_x = ori_data_path + "REFUGE1-Train-400/Images_comb"
    data_path_train_y = ori_data_path + "REFUGE1-Train-400/Masks"
    data_path_val_x   = ori_data_path + "REFUGE1-Val-400/REFUGE-Validation400"
    data_path_val_y   = ori_data_path + "REFUGE1-Val-400/REFUGE-Validation400-GT/Disc_Cup_Masks"
    data_path_test_x  = ori_data_path + "REFUGE1-Test-400/Images"
    data_path_test_y  = ori_data_path + "REFUGE1-Test-400/Masks"
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(data_path_train_x, data_path_train_y,
                                                                      data_path_val_x, data_path_val_y,
                                                                      data_path_test_x, data_path_test_y)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Val: {len(val_x)} - {len(val_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    create_dir("C:/Users/Josh/Desktop/4YP/Processed_Data/train/image/")
    create_dir("C:/Users/Josh/Desktop/4YP/Processed_Data/train/mask/")
    create_dir("C:/Users/Josh/Desktop/4YP/Processed_Data/val/image/")
    create_dir("C:/Users/Josh/Desktop/4YP/Processed_Data/val/mask/")
    create_dir("C:/Users/Josh/Desktop/4YP/Processed_Data/test/image/")
    create_dir("C:/Users/Josh/Desktop/4YP/Processed_Data/test/mask/")

    """ Data augmentation"""
    # set to True to increase training dataset, thus to recude overfitting
    augment_data(train_x, train_y, "C:/Users/Josh/Desktop/4YP/Processed_Data/train/", augment=True)
    #augment_data(val_x, val_y, "C:/Users/Josh/Desktop/4YP/Processed_Data/val/", augment=False)
    #augment_data(test_x, test_y, "C:/Users/Josh/Desktop/4YP/Processed_Data/test/", augment=False)
