import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PatchEmbed(nn.Module):
    """Audio to Patch Embedding for BEATs"""
    
    def __init__(self, 
                 img_size: Tuple[int, int] = (64, 1024),  # (freq, time)
                 patch_size: Tuple[int, int] = (8, 8),    # (freq_patch, time_patch)
                 in_chans: int = 1,
                 embed_dim: int = 768):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Ensure input size is divisible by patch size
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention for BEATs"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: [B, H, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # mask indicates which keys (columns) are valid
            # mask shape: [B, N] where N is sequence length
            # We need to mask out attention to invalid keys
            
            # Create attention mask: [B, 1, 1, N] -> [B, 1, N, N]
            # This will mask the keys (columns) for all queries
            key_mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
            key_mask = key_mask.expand(-1, -1, N, -1)  # [B, 1, N, N]
            
            # Apply mask to attention weights
            attn = attn.masked_fill(key_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer Block for BEATs"""
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class StandardBEATs(nn.Module):
    """BEATs model for standard log-mel spectrogram input (64 mel bands)"""
    
    def __init__(self,
                 img_size: Tuple[int, int] = (64, 1024),  # (freq, time)
                 patch_size: Tuple[int, int] = (8, 8),    # (freq_patch, time_patch)
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 output_size: int = 512):
        """
        Args:
            img_size: Size of input spectrogram (freq, time)
            patch_size: Size of each patch (freq_patch, time_patch)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate
            output_size: Final output feature size
        """
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # CLS token and dropout
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def _get_pos_embed(self, H, W, device):
        """Get positional embedding for given height and width"""
        # Calculate grid size based on actual input size
        grid_H = math.ceil(H / self.patch_size[0])
        grid_W = math.ceil(W / self.patch_size[1])
        num_patches = grid_H * grid_W
        
        # Create positional embedding on the fly
        pos_embed = torch.zeros(1, num_patches + 1, self.embed_dim, device=device)
        
        # Simple positional encoding (sinusoidal)
        pe = torch.zeros(num_patches, self.embed_dim)
        position = torch.arange(0, num_patches).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() *
                           -(math.log(10000.0) / self.embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pos_embed[0, 1:] = pe.to(device)
        # CLS token position is kept as zeros
        
        return pos_embed
    
    def forward_features(self, x, mask: Optional[torch.Tensor] = None):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        num_patches = x.shape[1]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        total_length = x.shape[1]  # num_patches + 1
        
        # Get dynamic positional embedding
        pos_embed = self._get_pos_embed(H, W, x.device)
        
        # Add positional encoding
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Create attention mask if needed
        if mask is not None:
            # The mask applies to the original time sequence
            # We need to downsample it to match the number of patches
            
            # Calculate how many time patches we have
            time_patches = math.ceil(W / self.patch_size[1])
            freq_patches = math.ceil(H / self.patch_size[0]) 
            expected_patches = time_patches * freq_patches
            
            # Debug prints to understand the issue
            # print(f"Input: {H}x{W}, Patches: {freq_patches}x{time_patches} = {expected_patches}, Actual: {num_patches}")
            # print(f"Mask input: {mask.shape}, Total length with CLS: {total_length}")
            
            # Downsample mask to match time patches
            if mask.shape[1] != time_patches:
                mask_downsampled = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=time_patches,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
            else:
                mask_downsampled = mask.float()
            
            # Create full mask for all patches (assuming freq patches are always valid)
            # We replicate the time mask for all frequency patches
            patch_mask = mask_downsampled.unsqueeze(2).expand(-1, -1, freq_patches).reshape(B, -1)
            
            # Ensure the patch mask matches the actual number of patches
            if patch_mask.shape[1] != num_patches:
                patch_mask = patch_mask[:, :num_patches]
            
            # Add cls token to mask (cls token is always attended to)
            cls_mask = torch.ones(B, 1, device=mask.device)
            full_mask = torch.cat((cls_mask, patch_mask), dim=1)
            
            # Ensure full_mask has the right shape
            assert full_mask.shape[1] == total_length, f"Mask length {full_mask.shape[1]} != total length {total_length}"
        else:
            full_mask = None
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, full_mask)
        
        x = self.norm(x)
        return x
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input spectrogram [batch, 1, mel_bands, time]
            mask: Optional mask [batch, time]
        Returns:
            features: Output features [batch, output_size]
        """
        x = self.forward_features(x, mask)
        
        # Use cls token for classification
        cls_token = x[:, 0]
        output = self.head(cls_token)
        
        return output


class CNNFrontend(nn.Module):
    """CNN frontend to process multiscale input for BEATs"""
    
    def __init__(self,
                 input_mel_bands: int = 192,
                 output_mel_bands: int = 64,
                 channels: list = [64, 128, 256]):
        super().__init__()
        
        self.input_mel_bands = input_mel_bands
        self.output_mel_bands = output_mel_bands
        
        # Calculate required downsampling
        self.downsample_factor = input_mel_bands // output_mel_bands
        
        layers = []
        in_channels = 1
        
        # Add convolutional layers
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        
        # Add final layer to match output dimensions
        layers.extend([
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((output_mel_bands, None))  # Downsample freq to target size
        ])
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Input multiscale spectrogram [batch, 1, 192, time]
        Returns:
            x: Downsampled spectrogram [batch, 1, 64, time]
        """
        return self.layers(x)


class MultiscaleBEATs(nn.Module):
    """BEATs model with CNN frontend for multiscale input"""
    
    def __init__(self,
                 input_mel_bands: int = 192,
                 output_mel_bands: int = 64,
                 img_size: Tuple[int, int] = (64, 1024),  # (freq, time)
                 patch_size: Tuple[int, int] = (8, 8),    # (freq_patch, time_patch)
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 output_size: int = 512):
        """
        Args:
            input_mel_bands: Number of mel bands in multiscale input
            output_mel_bands: Number of mel bands after CNN processing
            img_size: Size of input spectrogram after CNN (freq, time)
            patch_size: Size of each patch (freq_patch, time_patch)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout rate
            output_size: Final output feature size
        """
        super().__init__()
        
        # CNN frontend to process multiscale input
        self.cnn_frontend = CNNFrontend(input_mel_bands, output_mel_bands)
        
        # BEATs encoder
        self.beats = StandardBEATs(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            output_size=output_size
        )
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input multiscale spectrogram [batch, 1, 192, time]
            mask: Optional mask [batch, time]
        Returns:
            features: Output features [batch, output_size]
        """
        # Process multiscale input through CNN
        x_processed = self.cnn_frontend(x)  # [batch, 1, 64, time]
        
        # Forward through BEATs
        return self.beats(x_processed, mask)