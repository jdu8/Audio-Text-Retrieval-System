import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardCRNN(nn.Module):
    """CRNN model for standard log-mel spectrogram input (64 mel bands)"""
    
    def __init__(self, 
                 input_mel_bands: int = 64,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 output_size: int = 512,
                 dropout: float = 0.3):
        """
        Args:
            input_mel_bands: Number of mel bands in input spectrogram (64 for standard)
            hidden_size: Hidden size for LSTM layers
            num_layers: Number of LSTM layers
            output_size: Final output feature size
            dropout: Dropout rate
        """
        super(StandardCRNN, self).__init__()
        
        self.input_mel_bands = input_mel_bands
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Only reduces freq by 2, preserves time
            nn.Dropout2d(dropout),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Only reduces freq by 2, preserves time
            nn.Dropout2d(dropout),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Only reduces freq by 2, preserves time
            nn.Dropout2d(dropout),
        )
        
        # Calculate the size after conv layers
        # 64 -> 32 -> 16 -> 8 (after 3 maxpool with stride 2)
        conv_output_size = 256 * (input_mel_bands // 8)
        
        # Recurrent layers
        self.rnn = nn.LSTM(
            input_size=conv_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Final projection layer
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        # Global average pooling for sequence aggregation
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input spectrogram [batch, 1, mel_bands, time]
            mask: Optional mask [batch, time] (not used in this basic version)
        Returns:
            features: Output features [batch, output_size]
        """
        batch_size, channels, freq, time = x.shape
        
        # Convolutional feature extraction
        conv_features = self.conv_layers(x)  # [batch, 256, freq/8, time]
        
        # Reshape for RNN: [batch, time, features]
        conv_features = conv_features.permute(0, 3, 1, 2)  # [batch, time, 256, freq/8]
        conv_features = conv_features.contiguous().view(batch_size, time, -1)
        
        # RNN processing
        rnn_output, _ = self.rnn(conv_features)  # [batch, time, hidden_size*2]
        
        # Global average pooling across time dimension
        rnn_output = rnn_output.permute(0, 2, 1)  # [batch, hidden_size*2, time]
        pooled_output = self.global_avg_pool(rnn_output).squeeze(-1)  # [batch, hidden_size*2]
        
        # Final projection
        features = self.projection(pooled_output)  # [batch, output_size]
        
        return features


class MultiscaleCRNN(nn.Module):
    """CRNN model for multiscale log-mel spectrogram input (192 mel bands)"""
    
    def __init__(self, 
                 input_mel_bands: int = 192,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 output_size: int = 512,
                 dropout: float = 0.3):
        """
        Args:
            input_mel_bands: Number of mel bands in input spectrogram (192 for multiscale)
            hidden_size: Hidden size for LSTM layers
            num_layers: Number of LSTM layers
            output_size: Final output feature size
            dropout: Dropout rate
        """
        super(MultiscaleCRNN, self).__init__()
        
        self.input_mel_bands = input_mel_bands
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Convolutional layers for feature extraction
        # Need more layers due to larger input size
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Only reduces freq by 2, preserves time
            nn.Dropout2d(dropout),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Only reduces freq by 2, preserves time
            nn.Dropout2d(dropout),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Only reduces freq by 2, preserves time
            nn.Dropout2d(dropout),
            
            # Fourth conv block (additional for multiscale)
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Only reduces freq by 2, preserves time
            nn.Dropout2d(dropout),
        )
        
        # Calculate the size after conv layers
        # 192 -> 96 -> 48 -> 24 -> 12 (after 4 maxpool with stride 2)
        conv_output_size = 512 * (input_mel_bands // 16)
        
        # Recurrent layers
        self.rnn = nn.LSTM(
            input_size=conv_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Final projection layer
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        # Global average pooling for sequence aggregation
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input spectrogram [batch, 1, mel_bands, time]
            mask: Optional mask [batch, time] (not used in this basic version)
        Returns:
            features: Output features [batch, output_size]
        """
        batch_size, channels, freq, time = x.shape
        
        # Convolutional feature extraction
        conv_features = self.conv_layers(x)  # [batch, 512, freq/16, time]
        
        # Reshape for RNN: [batch, time, features]
        conv_features = conv_features.permute(0, 3, 1, 2)  # [batch, time, 512, freq/16]
        conv_features = conv_features.contiguous().view(batch_size, time, -1)
        
        # RNN processing
        rnn_output, _ = self.rnn(conv_features)  # [batch, time, hidden_size*2]
        
        # Global average pooling across time dimension
        rnn_output = rnn_output.permute(0, 2, 1)  # [batch, hidden_size*2, time]
        pooled_output = self.global_avg_pool(rnn_output).squeeze(-1)  # [batch, hidden_size*2]
        
        # Final projection
        features = self.projection(pooled_output)  # [batch, output_size]
        
        return features


