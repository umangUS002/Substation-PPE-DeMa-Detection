import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class SpatialFeatureExtractor(nn.Module):
    """
    Spatial Path: Extracts candidate regions from intermediate feature maps.
    In an FPN, these would be P4 and P5. For a VGG-16 SSD backbone, 
    we extract intermediate feature layers (e.g., Conv4_3 and Conv7) 
    to serve as our spatial candidate proposals.
    """
    def __init__(self, pretrained_backbone=None):
        super(SpatialFeatureExtractor, self).__init__()
        # Use torchvision's VGG16 features if no backbone provided
        if pretrained_backbone is None:
            vgg = torchvision.models.vgg16(pretrained=True)
            self.features = vgg.features
        else:
            self.features = pretrained_backbone

    def forward(self, x):
        """
        Passes the image through the backbone and returns intermediate 
        feature maps acting as P4 and P5 equivalents.
        """
        p4_equivalent = None
        p5_equivalent = None
        
        # We manually iterate to grab specific intermediate layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 22: # Approx Conv4_3 in VGG16 (acting as P4)
                p4_equivalent = x
            elif i == 29: # Maxpool5 output (acting as P5)
                p5_equivalent = x
                break
                
        return p4_equivalent, p5_equivalent

class Combustion3DClassifier(nn.Module):
    """
    Temporal Path 3D Convolutional Classifier for Fire and Ash Detection.
    Expects input shape: (Batch, Channels, Time, Height, Width)
    Where Time = 16 frames.
    
    Examines:
    - Flicker frequency
    - Irregular boundary turbulence
    - Directional growth/acceleration
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super(Combustion3DClassifier, self).__init__()
        # num_classes: 0 = Background/Steam, 1 = Combustion (Fire/Ash)
        
        # Spatial-Temporal Convolution Block 1
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Spatial-Temporal Convolution Block 2
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Spatial-Temporal Convolution Block 3
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Adaptive pooling to handle varying candidate region sizes, outputting a fixed spatial size (2x2)
        # Time dimension is reduced to 2 (16 -> 8 -> 4 -> 2)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        # Fully connected decision layers
        self.fc1 = nn.Linear(128 * 2 * 2 * 2, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x shape: (Batch, Channels, 16, H, W)
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
