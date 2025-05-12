class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)), # Change to Conv2d
            nn.BatchNorm2d(32), # Change to BatchNorm2d
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout here
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)), # Change to Conv2d, Adjust kernel and padding
            nn.BatchNorm2d(64), # Change to BatchNorm2d
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout here
            nn.MaxPool2d((2, 2)), # Change to MaxPool2d, adjust kernel
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)), # Change to Conv2d, adjust kernel and padding
            nn.BatchNorm2d(128), # Change to BatchNorm2d
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Change to AdaptiveAvgPool2d, adjust output
        )
        self.proj = nn.Linear(128, 768)
        self.beats = AutoModel.from_pretrained('facebook/wav2vec2-base-960h')

    def forward(self, mel_specs):
        x = self.cnn(mel_specs).squeeze()
        x = self.proj(x)
        # Return the projected CNN output directly
        return x
