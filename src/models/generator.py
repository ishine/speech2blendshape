import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipModule(nn.Module):
    def __init__(self, min_threshold, max_threshold):
        super(ClipModule, self).__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    
    def forward(self, x):
        return torch.clamp(x, min=self.min_threshold, max=self.max_threshold)


class FCGenerator(nn.Module):
    def __init__(self, fc1_dim, fc2_dim):
        super(FCGenerator, self).__init__()

        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim

        fully_connected = nn.Sequential(
            nn.AdaptiveAvgPool1d(self.fc1_dim),
            nn.Linear(self.fc1_dim, self.fc2_dim, bias=False),
            nn.ReLU6(),
            nn.Dropout(0.2),
            nn.Linear(self.fc2_dim, 16, bias=False),
        )
        self.generator = nn.Sequential(
            fully_connected,
            # nn.Sigmoid()
            ClipModule(min_threshold=0, max_threshold=1),
        )

    def forward(self, x):
        return self.generator(x)


class CNNGenerator(nn.Module):
    def __init__(self, frame_window=16, attention_window=8, num_classes=16):
        super(CNNGenerator, self).__init__()
        self.frame_window = frame_window
        self.attention_window = attention_window
        self.num_classes = num_classes
        
        self.frame_conv = nn.Sequential(
            nn.Conv2d(29, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), #  29 x 16 x 1 => 32 x 8 x 1
            nn.LeakyReLU(0.02, True),
            nn.Conv2d(32, 32, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 32 x 8 x 1 => 32 x 4 x 1
            nn.LeakyReLU(0.02, True),
            nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 32 x 4 x 1 => 64 x 2 x 1
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=True), # 64 x 2 x 1 => 64 x 1 x 1
            nn.LeakyReLU(0.2, True),
        )
        self.frame_fc = nn.Sequential(
            nn.Linear(in_features=64, out_features=128, bias=True),
            nn.LeakyReLU(0.02),

            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.LeakyReLU(0.02),            

            nn.Linear(in_features=64, out_features=32, bias=True),          
            nn.Tanh()
            )

        self.attention_conv = nn.Sequential( # b x subspace_dim x seq_len
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attention = nn.Sequential(
            nn.Linear(self.attention_window, self.attention_window, bias=True),   
            nn.Softmax(dim=1)
            )

        self.fc = nn.Sequential(
            nn.Linear(32, self.num_classes, bias=False),
            ClipModule(0, 1)
        )


    def forward(self, speech_features): # B, 29, T
        # Per-Frame Audio Feature Estimation
        windowed_features = self.frame_windowing(speech_features) # B, T, 29, frame_window
        B, T, C, W = windowed_features.shape
        windowed_features = windowed_features.view(B*T, C, W, 1) # B*T, C, W, 1
        frame_conv_out = self.frame_conv(windowed_features) # B*T, 64, 1
        frame_conv_out = frame_conv_out.view(B, T, -1) # B, T, 64
        frame_fc_out = self.frame_fc(frame_conv_out) # B, T, 32

        # Attention for smooth expression
        windowed_fc_out = self.attention_windowing(frame_fc_out) # B, T, 32, attention_window
        B, T, C, W = windowed_fc_out.shape
        windowed_fc_out = windowed_fc_out.view(B*T, C, W) # B*T, C, W
        att_conv_out = self.attention_conv(windowed_fc_out) # B*T, 1, W
        att_out = self.attention(att_conv_out) # B*T, 1, W
        att_out = att_out.permute(0, 2, 1) # B*T, W, 1

        final_features = torch.bmm(windowed_fc_out, att_out) # B*T, C, 1
        final_features = final_features.view(B, T, C) # B, T, C

        # feature -> blendshape
        predictions = self.fc(final_features) # B, T, num_classes

        return predictions


    def frame_windowing(self, speech_features):
        B, C, T = speech_features.shape
        W = self.frame_window
        padded_features = F.pad(speech_features, (W//2-1, W//2))
        output_features = torch.zeros((B, T, C, W)).to(speech_features.device)

        for seq in range(T):
            windowed_features = padded_features[:,:,seq:seq+W].unsqueeze(1)
            output_features[:,seq:seq+1,:,:] = windowed_features

        return output_features


    def attention_windowing(self, frame_fc_out):
        B, T, C = frame_fc_out.shape
        W = self.attention_window
        frame_fc_out = frame_fc_out.permute(0, 2, 1)
        padded_features = F.pad(frame_fc_out, (W//2-1, W//2))
        padded_features.permute(0, 2, 1)
        output_features = torch.zeros((B, T, C, W)).to(frame_fc_out.device)

        for seq in range(T):
            windowed_features = padded_features[:,:,seq:seq+W].unsqueeze(1)
            output_features[:,seq:seq+1,:,:] = windowed_features

        return output_features