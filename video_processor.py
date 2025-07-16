import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms, models
from tqdm import tqdm
import os

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = Conv3DBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class PretrainedResNet3D(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.in_channels = 64

        # Load pretrained ResNet18
        resnet2d = models.resnet18(pretrained=pretrained)

        # Initial convolution - inflate 2D kernel to 3D
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                              stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self._inflate_conv2d_to_3d(self.conv1, resnet2d.conv1)

        self.bn1 = nn.BatchNorm3d(64)
        self._copy_bn_params(self.bn1, resnet2d.bn1)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                  stride=(1, 2, 2), padding=(0, 1, 1))

        # ResNet layers
        self.layer1 = self._make_layer(64, 2, resnet2d.layer1)
        self.layer2 = self._make_layer(128, 2, resnet2d.layer2, stride=2)
        self.layer3 = self._make_layer(256, 2, resnet2d.layer3, stride=2)
        self.layer4 = self._make_layer(512, 2, resnet2d.layer4, stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Initialize FC layer
        if pretrained:
            self.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.fc.bias.data.zero_()

    def _make_layer(self, out_channels, blocks, pretrained_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample_conv = nn.Conv3d(self.in_channels, out_channels,
                                      kernel_size=1, stride=(1, stride, stride), bias=False)
            if hasattr(pretrained_layer[0], 'downsample'):
                self._inflate_conv2d_to_3d(downsample_conv, 
                                         pretrained_layer[0].downsample[0])

            downsample_bn = nn.BatchNorm3d(out_channels)
            if hasattr(pretrained_layer[0], 'downsample'):
                self._copy_bn_params(downsample_bn, 
                                   pretrained_layer[0].downsample[1])

            downsample = nn.Sequential(downsample_conv, downsample_bn)

        layers = []
        # First block with stride and downsample
        first_block = ResBlock3D(self.in_channels, out_channels, 
                               stride=(1, stride, stride), downsample=downsample)
        self._inflate_block(first_block, pretrained_layer[0])
        layers.append(first_block)

        # Remaining blocks
        self.in_channels = out_channels
        for i in range(1, blocks):
            block = ResBlock3D(self.in_channels, out_channels)
            self._inflate_block(block, pretrained_layer[i])
            layers.append(block)

        return nn.Sequential(*layers)

    def _inflate_conv2d_to_3d(self, conv3d, conv2d):
        """Inflate 2D convolution weights to 3D"""
        with torch.no_grad():
            # Inflate weight tensor: (C_out, C_in, H, W) -> (C_out, C_in, T, H, W)
            w3d = conv3d.weight.data
            w2d = conv2d.weight.data
            kernel_t = w3d.size(2)
            w3d.copy_(w2d.unsqueeze(2).repeat(1, 1, kernel_t, 1, 1) / kernel_t)

    def _copy_bn_params(self, bn3d, bn2d):
        """Copy BatchNorm parameters from 2D to 3D"""
        with torch.no_grad():
            bn3d.weight.data.copy_(bn2d.weight.data)
            bn3d.bias.data.copy_(bn2d.bias.data)
            bn3d.running_mean.copy_(bn2d.running_mean)
            bn3d.running_var.copy_(bn2d.running_var)

    def _inflate_block(self, block3d, block2d):
        """Inflate ResBlock from 2D to 3D"""
        # Inflate first convolution
        self._inflate_conv2d_to_3d(block3d.conv1.conv, block2d.conv1)
        self._copy_bn_params(block3d.conv1.bn, block2d.bn1)

        # Inflate second convolution
        self._inflate_conv2d_to_3d(block3d.conv2, block2d.conv2)
        self._copy_bn_params(block3d.bn2, block2d.bn2)

        # Inflate downsample if it exists
        if block3d.downsample is not None and block2d.downsample is not None:
            self._inflate_conv2d_to_3d(block3d.downsample[0], 
                                     block2d.downsample[0])
            self._copy_bn_params(block3d.downsample[1], 
                               block2d.downsample[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# In create_model function, add dropout and batch norm tuning:
def create_model(num_classes=2, pretrained=True):
    model = PretrainedResNet3D(num_classes=num_classes, pretrained=pretrained)
    return model

def apply_gaussian_blur(frame, intensity='maximum'):
    blur_params = {
        'low': ((15, 15), 5),
        'medium': ((25, 25), 10),
        'high': ((45, 45), 15),
        'very_high': ((99, 99), 30),
        'extreme': ((151, 151), 50),
        'maximum': ((299, 299), 150)
    }
    kernel_size, sigma = blur_params.get(intensity, blur_params['maximum'])
    return cv2.GaussianBlur(frame, kernel_size, sigma)

def process_video(video_path, output_path, model_path, progress_callback, threshold=0.40, batch_size=4, clip_length=96):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    progress_callback(f"Using device: {device}")
    
    try:
        # 1. Load model with matching structure from inference pipeline
        progress_callback("\nLoading model...")
        base_model = create_model(num_classes=2, pretrained=False)
        model = nn.Sequential(
            base_model,
            nn.Dropout(p=0.3)  # Match training dropout
        )
        
        # Load checkpoint with same handling as inference pipeline
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        try:
            model.load_state_dict(state_dict)
            progress_callback("Model loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading state dict: {str(e)}")

        model = model.to(device)
        model.eval()

        # 2. Setup video processing
        progress_callback("\nProcessing video...")
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),  # Match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Setup video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 3. Process frames into clips
        clips = []
        timestamps = []
        frames_buffer = []
        
        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Process remaining frames if we have any
                    if frames_buffer:
                        # Pad the remaining frames to make a full clip
                        while len(frames_buffer) < clip_length:
                            frames_buffer.append(frames_buffer[-1])  # Repeat last frame
                            
                        clip = torch.stack(frames_buffer)
                        clip = clip.permute(1, 0, 2, 3)  # [C, T, H, W]
                        clips.append(clip)
                        
                        start_time = (frame_idx - len(frames_buffer)) / fps
                        end_time = frame_idx / fps
                        timestamps.append((start_time, end_time))
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = transform(frame)
                frames_buffer.append(frame_tensor)
                
                if len(frames_buffer) == clip_length:
                    clip = torch.stack(frames_buffer)
                    clip = clip.permute(1, 0, 2, 3)  # [C, T, H, W]
                    clips.append(clip)
                    
                    start_time = (frame_idx - clip_length + 1) / fps
                    end_time = frame_idx / fps
                    timestamps.append((start_time, end_time))
                    
                    # 50% overlap
                    frames_buffer = frames_buffer[clip_length//2:]
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        # 4. Run inference
        progress_callback("\nRunning inference...")
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(clips), batch_size), desc="Inference"):
                batch_clips = clips[i:i + batch_size]
                batch_tensor = torch.stack(batch_clips).to(device)
                
                outputs = model(batch_tensor)
                probs = F.softmax(outputs, dim=1)
                
                preds = (probs[:, 1] > threshold).cpu().numpy()
                predictions.extend(preds)
        
        # 5. Generate filtered video
        progress_callback("\nGenerating visualization...")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        current_clip = 0
        
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Visualizing") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_count / fps
                
                # Find corresponding clip
                while (current_clip < len(timestamps) and 
                       current_time > timestamps[current_clip][1]):
                    current_clip += 1
                
                # Apply blur if we're in a clip with positive prediction
                if current_clip < len(predictions) and predictions[current_clip]:
                    frame = apply_gaussian_blur(frame)
                
                out.write(frame)
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        return {
            "success": True,
            "message": "Video content filtered successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()