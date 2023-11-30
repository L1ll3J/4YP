import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG19_Weights

from data_aug.data import train_test_split
from glob import glob
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import itertools

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100  # Update this as needed
learning_rate = 1e-4


class PairedContentStyleDataset(Dataset):
    def __init__(self, content_root_dir, style_dir, transform=None):
        content_root_dir = os.path.normpath(content_root_dir)
        subdirs = [os.path.join(content_root_dir, o) for o in os.listdir(content_root_dir)
                   if os.path.isdir(os.path.join(content_root_dir, o))]

        # List of all content images
        content_images = []
        for subdir in subdirs:
            content_subdir = os.path.normpath(os.path.join(subdir, 'images'))
            for f in glob(os.path.join(content_subdir, '*.png')):
                content_images.append(f)
        # List of all style images
        style_images = glob(os.path.join(style_dir, '*.jpg'))
        # Make sure we cycle through the style image
        max_num_images = max(len(content_images), len(style_images))
        self.content_images = list(itertools.islice(itertools.cycle(content_images), max_num_images)) if len(content_images) < max_num_images else content_images
        self.style_images = list(itertools.islice(itertools.cycle(style_images), max_num_images)) if len(style_images) < max_num_images else style_images
        # Number of samples is determined by the number of content images
        self.num_samples = max_num_images
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load and transform the content image
        content_path = self.content_images[idx]
        content_img = Image.open(content_path).convert('RGB')
        content_img = self.transform(content_img) if self.transform else content_img

        # Load and transform the next style image
        style_path = self.style_images[idx]
        style_img = Image.open(style_path).convert('RGB')
        style_img = self.transform(style_img) if self.transform else style_img

        return content_img, style_img

# Style Transfer Network
class AdaINStyleTransfer(nn.Module):
    def __init__(self, encoder, decoder):
        super(AdaINStyleTransfer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, content, style):
        content_features = self.encoder(content)
        style_features = self.encoder(style)
        t = self.adaptive_instance_normalization(content_features, style_features)
        g_t = self.decoder(t)
        return g_t

    def adaptive_instance_normalization(self, content_feat, style_feat):
        size = content_feat.size()
        style_mean, style_std = style_feat.mean([2, 3], keepdim=True), style_feat.std([2, 3], keepdim=True)
        content_mean, content_std = content_feat.mean([2, 3], keepdim=True), content_feat.std([2, 3], keepdim=True)
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

# Decoder mirrors the VGG-19 architecture but replaces pooling layers with nearest upsampling
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # Assuming VGG-19 up to relu4_1 has the following architecture:
        # conv3-64, conv3-64, pool, conv3-128, conv3-128, pool, conv3-256, conv3-256, conv3-256, conv3-256, pool, conv3-512, conv3-512, conv3-512, conv3-512
        # We will create a mirrored architecture with upsampling and convolutions

        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Repeat conv3-256 four times as in VGG-19 before the next upsampling
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Repeat conv3-128 twice as in VGG-19 before the next upsampling
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Repeat conv3-64 twice as in VGG-19 before the next upsampling
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Final layer to get to 3 channels for the image
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)  # Typically, no activation is used in the final layer, but paper specifics might vary
        )

    def forward(self, x):
        # Apply each layer set as defined above
        x = self.upconv4(x)
        x = self.conv4_3(x)
        x = self.upconv3(x)
        x = self.conv3_2(x)
        x = self.upconv2(x)
        x = self.conv2_2(x)
        x = self.final(x)
        return x


def gram_matrix(features):
    if len(features.size()) != 4:
        raise ValueError(f"Expected a 4D tensor, but got a tensor with shape {features.size()}")

    (b, ch, h, w) = features.size()
    features = features.view(b, ch, w * h)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram.div(ch * h * w)

def style_lossfn(output_features, style_features_vgg):
    loss = 0.0
    for fmap_output, fmap_style in zip(output_features, style_features_vgg):
        # Verify that each feature map is 3D (channels, height, width)
        if fmap_output.dim() != 3 or fmap_style.dim() != 3:
            raise ValueError("Feature maps should be 3D tensors.")

        # Add batch dimension to make them 4D
        fmap_output = fmap_output.unsqueeze(0)
        fmap_style = fmap_style.unsqueeze(0)

        output_gram = gram_matrix(fmap_output)
        style_gram = gram_matrix(fmap_style)
        loss += torch.nn.functional.mse_loss(output_gram, style_gram)
    return loss

# Load the VGG-19 model pre-trained on ImageNet data
vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:22].to(device).eval()  # Up to relu4_1

# Freeze VGG-19 parameters
for param in vgg19.parameters():
    param.requires_grad = False

image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

paired_dataset = PairedContentStyleDataset(content_root_dir='C:/Users/Josh/Desktop/4YP/Datasets/RAW/Kaggle/Unlabelled', style_dir='C:/Users/Josh/Desktop/4YP/Processed_Data/train/image', transform=image_transform)
paired_loader = DataLoader(paired_dataset, batch_size=8, shuffle=True)

# Initialize the AdaIN style transfer model
decoder = Decoder().to(device)  # Define your Decoder architecture
style_transfer_model = AdaINStyleTransfer(vgg19, decoder).to(device)

# Loss function and optimizer
mse_loss = nn.MSELoss()
optimizer = optim.Adam(style_transfer_model.decoder.parameters(), lr=learning_rate)

Lambda = [0, 0.25, 0.5, 0.75, 1.0]

for lambda_style in tqdm(Lambda):
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for content_images, style_images in paired_loader:
            content_images = content_images.to(device)
            style_images = style_images.to(device)
            optimizer.zero_grad()

            # Perform style transfer
            stylized_images = style_transfer_model(content_images, style_images)


            # Compute content loss using vgg19
            content_loss = mse_loss(vgg19(stylized_images), vgg19(content_images))

            # Compute style loss as the MSE between the Gram matrices of the style and output features
            style_loss = style_lossfn(vgg19(stylized_images), vgg19(style_images))

            # Backward pass and optimize
            total_loss = content_loss + lambda_style * style_loss
            total_loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}")

    # Save the trained decoder
    torch.save(style_transfer_model.decoder.state_dict(), f'decoder_lambda_{lambda_style}.pth')

# Note: The script is conceptual and assumes all placeholder functions and classes are defined.
# You need to replace placeholder code with actual implementations.