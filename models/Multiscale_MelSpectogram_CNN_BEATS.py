class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout after the first ReLU
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout after the second ReLU
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout after the third ReLU
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.proj = nn.Linear(128, 768)
        self.beats = AutoModel.from_pretrained('facebook/wav2vec2-base-960h')

    def forward(self, mel_specs):
        mel_specs = mel_specs.permute(0, 2, 1, 3, 4)
        x = self.cnn(mel_specs).squeeze()
        x = self.proj(x)
        # Return the projected CNN output directly
        return x
