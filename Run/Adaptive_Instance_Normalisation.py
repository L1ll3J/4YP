import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_aug.data import train_test_split
from utils import *
import torch.optim as optim
import wandb
from torchvision.models import vgg19
from glob import glob

# STYLE NETWORK
def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = style_feat.mean([2, 3]), style_feat.std([2, 3])
    content_mean, content_std = content_feat.mean([2, 3]), content_feat.std([2, 3])

    normalized_feat = (content_feat - content_mean[None, :, None, None]) / content_std[None, :, None, None]
    stylized_feat = normalized_feat * style_std[None, :, None, None] + style_mean[None, :, None, None]

    return stylized_feat


# Define the Encoder and Decoder network
class StyleTransferNetwork(nn.Module):
    def __init__(self):
        super(StyleTransferNetwork, self).__init__()
        vgg = vgg19(pretrained=True).features
        # Assume we have a function 'vgg_layers' that extracts the first few layers
        self.encoder = vgg_layers(vgg, layer_name='relu4_1')
        # Load the decoder model (manually transfer the weights from the .t7 file)
        self.decoder = Decoder()  # You'll need to define this class

    def forward(self, content_img, style_img):
        content_features = self.encoder(content_img)
        style_features = self.encoder(style_img)
        t = adaptive_instance_normalization(content_features, style_features)
        g_t = self.decoder(t)
        return g_t


# Define the function to extract the VGG layers
def vgg_layers(vgg, layer_name):
    layers = []
    for name, layer in vgg._modules.items():
        layers.append(layer)
        if name == layer_name:
            break
    return nn.Sequential(*layers)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Assuming VGG19 up to relu4_1 has 8 layers, we'll define a mirrored decoder.
        # This needs to be adapted based on the actual layers of your encoder.
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # Corresponds to the last VGG conv layer
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(64, 3, kernel_size=3, padding=1)  # Output layer

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Upsample and apply convolution
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # Corresponds to unpooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv8(x)  # No ReLU for the final output layer
        return x

    def _initialize_weights(self):
        # Weight initialization logic (if any) goes here, typically using Xavier or Kaiming initialization.
        # For example:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    style_transfer_model = StyleTransferNetwork().to(device)

    train_x = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/train/image/*"))
    train_y = sorted(glob("C:/Users/Josh/Desktop/4YP/Processed_Data/train/mask/*"))

    train_dataset = train_test_split(train_x, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=1,
                              pin_memory=True)

    unlabelled_dataloaders = []
    for unlabelled_path in sorted(glob("filepath/*/image")):
        unlabelled_dataset = train_test_split(unlabelled_path)  # You will need to implement CustomDataset
        unlabelled_loader = DataLoader(dataset=unlabelled_dataset, batch_size=16, shuffle=False)
        unlabelled_dataloaders.append(unlabelled_loader)
