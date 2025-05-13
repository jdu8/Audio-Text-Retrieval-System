import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SpectrogramConvBlock(nn.Module):
    """Convolutional block for spectrogram processing"""
    
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), 
                 padding=(1, 1), pool_kernel=(2, 1), dropout=0.2):
        super(SpectrogramConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_kernel, pool_kernel)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class TemporalConvBlock(nn.Module):
    """1D Convolutional block for temporal feature processing"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dropout=0.2):
        super(TemporalConvBlock, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class StandardSpec2Vec(nn.Module):
    """Spec2Vec model for standard log-mel spectrogram input (64 mel bands)"""
    
    def __init__(self, 
                 input_mel_bands: int = 64,
                 conv_channels: List[int] = [32, 64, 128, 256],
                 temporal_channels: List[int] = [256, 512],
                 output_size: int = 512,
                 dropout: float = 0.2):
        """
        Args:
            input_mel_bands: Number of mel bands in input spectrogram (64 for standard)
            conv_channels: Channel sizes for 2D convolutional layers
            temporal_channels: Channel sizes for 1D temporal processing
            output_size: Final output feature size
            dropout: Dropout rate
        """
        super(StandardSpec2Vec, self).__init__()
        
        self.input_mel_bands = input_mel_bands
        self.output_size = output_size
        
        # 2D Convolutional layers for spectral feature extraction
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in conv_channels:
            self.conv_layers.append(
                SpectrogramConvBlock(
                    in_channels, out_channels, 
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                    pool_kernel=(2, 1), dropout=dropout
                )
            )
            in_channels = out_channels
        
        # Calculate size after conv layers
        # 64 -> 32 -> 16 -> 8 -> 4 (4 maxpool layers with stride 2 in freq dimension)
        conv_output_freq_size = input_mel_bands // (2 ** len(conv_channels))
        conv_output_size = conv_channels[-1] * conv_output_freq_size
        
        # Temporal processing with 1D convolutions
        self.temporal_layers = nn.ModuleList()
        in_channels = conv_output_size
        
        for out_channels in temporal_channels:
            self.temporal_layers.append(
                TemporalConvBlock(in_channels, out_channels, kernel_size=3, dropout=dropout)
            )
            in_channels = out_channels
        
        # Global pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_projection = nn.Sequential(
            nn.Linear(temporal_channels[-1], temporal_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(temporal_channels[-1] // 2, output_size)
        )
        
        # Initialize attention mechanism for weighted temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(temporal_channels[-1], temporal_channels[-1] // 4),
            nn.ReLU(inplace=True),
            nn.Linear(temporal_channels[-1] // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input spectrogram [batch, 1, mel_bands, time]
            mask: Optional mask [batch, time]
        Returns:
            features: Output features [batch, output_size]
        """
        batch_size, channels, freq, time = x.shape
        
        # Apply 2D convolutional layers
        conv_features = x
        for conv_layer in self.conv_layers:
            conv_features = conv_layer(conv_features)
        
        # Reshape for temporal processing
        conv_features = conv_features.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
        conv_features = conv_features.contiguous().view(batch_size, time, -1)
        conv_features = conv_features.permute(0, 2, 1)  # [batch, features, time]
        
        # Apply temporal layers
        temporal_features = conv_features
        for temporal_layer in self.temporal_layers:
            temporal_features = temporal_layer(temporal_features)
        
        # Apply attention-weighted pooling
        temporal_features_t = temporal_features.permute(0, 2, 1)  # [batch, time, features]
        attention_weights = self.attention(temporal_features_t)  # [batch, time, 1]
        
        # Apply mask if provided
        if mask is not None:
            # Downsample mask to match temporal features
            mask_downsampled = F.interpolate(
                mask.unsqueeze(1).float(),
                size=temporal_features_t.shape[1],
                mode='linear',
                align_corners=False
            ).squeeze(1)
            attention_weights = attention_weights.squeeze(-1) * mask_downsampled.unsqueeze(-1)
            attention_weights = attention_weights.unsqueeze(-1)
        
        # Weighted pooling
        weighted_features = (temporal_features_t * attention_weights).sum(dim=1)  # [batch, features]
        
        # Final projection
        output = self.final_projection(weighted_features)
        
        return output


class MultiscaleSpec2Vec(nn.Module):
    """Spec2Vec model for multiscale log-mel spectrogram input (192 mel bands)"""
    
    def __init__(self, 
                 input_mel_bands: int = 192,
                 conv_channels: List[int] = [64, 128, 256, 512, 1024],
                 temporal_channels: List[int] = [512, 1024],
                 output_size: int = 512,
                 dropout: float = 0.2):
        """
        Args:
            input_mel_bands: Number of mel bands in input spectrogram (192 for multiscale)
            conv_channels: Channel sizes for 2D convolutional layers
            temporal_channels: Channel sizes for 1D temporal processing
            output_size: Final output feature size
            dropout: Dropout rate
        """
        super(MultiscaleSpec2Vec, self).__init__()
        
        self.input_mel_bands = input_mel_bands
        self.output_size = output_size
        
        # 2D Convolutional layers for spectral feature extraction
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for out_channels in conv_channels:
            self.conv_layers.append(
                SpectrogramConvBlock(
                    in_channels, out_channels, 
                    kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                    pool_kernel=(2, 1), dropout=dropout
                )
            )
            in_channels = out_channels
        
        # Calculate size after conv layers
        # 192 -> 96 -> 48 -> 24 -> 12 -> 6 (5 maxpool layers with stride 2 in freq dimension)
        conv_output_freq_size = input_mel_bands // (2 ** len(conv_channels))
        conv_output_size = conv_channels[-1] * conv_output_freq_size
        
        # Temporal processing with 1D convolutions
        self.temporal_layers = nn.ModuleList()
        in_channels = conv_output_size
        
        for out_channels in temporal_channels:
            self.temporal_layers.append(
                TemporalConvBlock(in_channels, out_channels, kernel_size=3, dropout=dropout)
            )
            in_channels = out_channels
        
        # Global pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_projection = nn.Sequential(
            nn.Linear(temporal_channels[-1], temporal_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(temporal_channels[-1] // 2, output_size)
        )
        
        # Initialize attention mechanism for weighted temporal pooling
        self.attention = nn.Sequential(
            nn.Linear(temporal_channels[-1], temporal_channels[-1] // 4),
            nn.ReLU(inplace=True),
            nn.Linear(temporal_channels[-1] // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input spectrogram [batch, 1, mel_bands, time]
            mask: Optional mask [batch, time]
        Returns:
            features: Output features [batch, output_size]
        """
        batch_size, channels, freq, time = x.shape
        
        # Apply 2D convolutional layers
        conv_features = x
        for conv_layer in self.conv_layers:
            conv_features = conv_layer(conv_features)
        
        # Reshape for temporal processing
        conv_features = conv_features.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
        conv_features = conv_features.contiguous().view(batch_size, time, -1)
        conv_features = conv_features.permute(0, 2, 1)  # [batch, features, time]
        
        # Apply temporal layers
        temporal_features = conv_features
        for temporal_layer in self.temporal_layers:
            temporal_features = temporal_layer(temporal_features)
        
        # Apply attention-weighted pooling
        temporal_features_t = temporal_features.permute(0, 2, 1)  # [batch, time, features]
        attention_weights = self.attention(temporal_features_t)  # [batch, time, 1]
        
        # Apply mask if provided
        if mask is not None:
            # Downsample mask to match temporal features
            mask_downsampled = F.interpolate(
                mask.unsqueeze(1).float(),
                size=temporal_features_t.shape[1],
                mode='linear',
                align_corners=False
            ).squeeze(1)
            attention_weights = attention_weights.squeeze(-1) * mask_downsampled.unsqueeze(-1)
            attention_weights = attention_weights.unsqueeze(-1)
        
        # Weighted pooling
        weighted_features = (temporal_features_t * attention_weights).sum(dim=1)  # [batch, features]
        
        # Final projection
        output = self.final_projection(weighted_features)
        
        return output