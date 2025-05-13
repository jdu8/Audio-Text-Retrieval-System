import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class TCNBlock(nn.Module):
    """A single TCN block with dilated convolution and residual connection"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout):
        super(TCNBlock, self).__init__()
        
        # Dilated convolution with padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        
        # First conv layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second conv layer
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        if self.residual_conv:
            residual = self.residual_conv(residual)
        
        out += residual
        out = self.relu2(out)
        out = self.dropout2(out)
        
        return out


class StandardTCN(nn.Module):
    """TCN model for standard log-mel spectrogram input (64 mel bands)"""
    
    def __init__(self, 
                 input_mel_bands: int = 64,
                 channel_sizes: List[int] = [64, 128, 256, 512],
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 output_size: int = 512):
        """
        Args:
            input_mel_bands: Number of mel bands in input spectrogram (64 for standard)
            channel_sizes: List of channel sizes for each TCN block
            kernel_size: Kernel size for dilated convolutions
            dropout: Dropout rate
            output_size: Final output feature size
        """
        super(StandardTCN, self).__init__()
        
        self.input_mel_bands = input_mel_bands
        self.output_size = output_size
        
        # Initial convolution to reduce frequency dimension
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce freq by 2, preserve time
            nn.Dropout2d(dropout),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce freq by 2, preserve time
            nn.Dropout2d(dropout),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce freq by 2, preserve time
        )
        
        # Calculate size after initial convolution: 64 -> 32 -> 16 -> 8
        reduced_freq_size = input_mel_bands // 8
        tcn_input_size = 256 * reduced_freq_size
        
        # TCN layers
        self.tcn_layers = nn.ModuleList()
        
        # Add initial projection if needed
        if tcn_input_size != channel_sizes[0]:
            self.input_projection = nn.Conv1d(tcn_input_size, channel_sizes[0], 1)
        else:
            self.input_projection = None
        
        # Create TCN blocks with increasing dilation
        in_channels = channel_sizes[0]
        for i, out_channels in enumerate(channel_sizes):
            dilation = 2 ** i  # Exponentially increasing dilation
            self.tcn_layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, 1, dilation, dropout)
            )
            in_channels = out_channels
        
        # Final projection layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(channel_sizes[-1], channel_sizes[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channel_sizes[-1] // 2, output_size)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input spectrogram [batch, 1, mel_bands, time]
            mask: Optional mask [batch, time] (not used in this basic version)
        Returns:
            features: Output features [batch, output_size]
        """
        batch_size, channels, freq, time = x.shape
        
        # Initial 2D convolution to reduce frequency dimension
        conv_features = self.initial_conv(x)  # [batch, 256, freq/8, time]
        
        # Reshape for 1D convolution: combine channel and freq dimensions
        conv_features = conv_features.permute(0, 3, 1, 2)  # [batch, time, 256, freq/8]
        conv_features = conv_features.contiguous().view(batch_size, time, -1)
        conv_features = conv_features.permute(0, 2, 1)  # [batch, features, time]
        
        # Apply input projection if needed
        if self.input_projection:
            conv_features = self.input_projection(conv_features)
        
        # Apply TCN blocks
        tcn_output = conv_features
        for tcn_layer in self.tcn_layers:
            tcn_output = tcn_layer(tcn_output)
        
        # Final classification
        features = self.classifier(tcn_output)
        
        return features


class MultiscaleTCN(nn.Module):
    """TCN model for multiscale log-mel spectrogram input (192 mel bands)"""
    
    def __init__(self, 
                 input_mel_bands: int = 192,
                 channel_sizes: List[int] = [128, 256, 512, 1024],
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 output_size: int = 512):
        """
        Args:
            input_mel_bands: Number of mel bands in input spectrogram (192 for multiscale)
            channel_sizes: List of channel sizes for each TCN block
            kernel_size: Kernel size for dilated convolutions
            dropout: Dropout rate
            output_size: Final output feature size
        """
        super(MultiscaleTCN, self).__init__()
        
        self.input_mel_bands = input_mel_bands
        self.output_size = output_size
        
        # Initial convolution to reduce frequency dimension (more layers for larger input)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce freq by 2, preserve time
            nn.Dropout2d(dropout),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce freq by 2, preserve time
            nn.Dropout2d(dropout),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce freq by 2, preserve time
            nn.Dropout2d(dropout),
            
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce freq by 2, preserve time
        )
        
        # Calculate size after initial convolution: 192 -> 96 -> 48 -> 24 -> 12
        reduced_freq_size = input_mel_bands // 16
        tcn_input_size = 512 * reduced_freq_size
        
        # TCN layers
        self.tcn_layers = nn.ModuleList()
        
        # Add initial projection if needed
        if tcn_input_size != channel_sizes[0]:
            self.input_projection = nn.Conv1d(tcn_input_size, channel_sizes[0], 1)
        else:
            self.input_projection = None
        
        # Create TCN blocks with increasing dilation
        in_channels = channel_sizes[0]
        for i, out_channels in enumerate(channel_sizes):
            dilation = 2 ** i  # Exponentially increasing dilation
            self.tcn_layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, 1, dilation, dropout)
            )
            in_channels = out_channels
        
        # Final projection layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(channel_sizes[-1], channel_sizes[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channel_sizes[-1] // 2, output_size)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input spectrogram [batch, 1, mel_bands, time]
            mask: Optional mask [batch, time] (not used in this basic version)
        Returns:
            features: Output features [batch, output_size]
        """
        batch_size, channels, freq, time = x.shape
        
        # Initial 2D convolution to reduce frequency dimension
        conv_features = self.initial_conv(x)  # [batch, 512, freq/16, time]
        
        # Reshape for 1D convolution: combine channel and freq dimensions
        conv_features = conv_features.permute(0, 3, 1, 2)  # [batch, time, 512, freq/16]
        conv_features = conv_features.contiguous().view(batch_size, time, -1)
        conv_features = conv_features.permute(0, 2, 1)  # [batch, features, time]
        
        # Apply input projection if needed
        if self.input_projection:
            conv_features = self.input_projection(conv_features)
        
        # Apply TCN blocks
        tcn_output = conv_features
        for tcn_layer in self.tcn_layers:
            tcn_output = tcn_layer(tcn_output)
        
        # Final classification
        features = self.classifier(tcn_output)
        
        return features