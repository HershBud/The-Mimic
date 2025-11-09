'''
Hi!
I'm Hersh
Glad you're here!
I made this model to adjust synthesizer parameters based on an input sound clip using a PyTorch NN model
'''

import config

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import parameters from your config file or define them here
# Example values for a typical 1.0s clip at 22050Hz:
N_MELS = 128
INPUT_TIME_FRAMES = 95
LATENT_DIM = 64


class AudioAutoencoder(nn.Module):
    def __init__(self, n_mels, time_frames, latent_dim):
        super().__init__()
        
        # NOTE: Input spectrogram shape is (Batch, 1, n_mels, time_frames)
        self.n_mels = n_mels
        self.time_frames = time_frames
        self.latent_dim = latent_dim
        
        # --- ENCODER: Compresses the spectrogram ---
        self.encoder = nn.Sequential(
            # Layer 1: Downsample H and W by a factor of 2. Channels: 1 -> 32
            # Output: [B, 32, n_mels/2, time_frames/2]
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # Layer 2: Downsample H and W by a factor of 2. Channels: 32 -> 64
            # Output: [B, 64, n_mels/4, time_frames/4]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Layer 3: Further compress. Channels: 64 -> 128
            # Output: [B, 128, n_mels/8, time_frames/8]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
            # YOU MAY ADD MORE CONV2D LAYERS HERE IF NEEDED. REMEMBER TO TRACK THE FINAL SPATIAL SIZE.
        )
        
        # Calculate the size of the tensor BEFORE it's flattened for the linear layer
        # For simplicity, we assume n_mels and time_frames are divisible by 8.
        self.final_conv_channels = 128
        self.final_h = n_mels // 8
        self.final_w = time_frames // 8
        self.linear_input_size = self.final_conv_channels * self.final_h * self.final_w
        
        # Linear layer to produce the latent vector
        self.fc_encode = nn.Linear(self.linear_input_size, self.latent_dim)
        
        # --- DECODER: Reconstructs the spectrogram ---
        
        # Linear layer to expand the latent vector back to the flattened 3D size
        self.fc_decode = nn.Linear(self.latent_dim, self.linear_input_size)
        
        self.decoder = nn.Sequential(
            # Layer 1: Upsample H and W by a factor of 2. Channels: 128 -> 64
            # Input is the reshaped tensor [B, 128, final_h, final_w]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0), 
            # OUTPUT_PADDING IS CRITICAL TO ENSURE CORRECT SHAPE REVERSAL. SET TO 1 IF CONV2D STRIDE IS 2.
            nn.LeakyReLU(0.2),

            # Layer 2: Upsample H and W by a factor of 2. Channels: 64 -> 32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0), 
            nn.LeakyReLU(0.2),

            # Layer 3: Final upsample H and W by a factor of 2. Channels: 32 -> 1
            # Output should be [B, 1, n_mels, time_frames]
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=0), 
            nn.Sigmoid() # Sigmoid is good for normalized spectrograms (0 to 1)
        )

    def forward(self, x):
        # Store the original batch size for later reshaping
        batch_size = x.size(0)
        
        # --- ENCODER FORWARD PASS ---
        x = self.encoder(x)
        
        # Flatten the spatial dimensions: [B, C, H, W] -> [B, C*H*W]
        x = x.view(batch_size, -1)
        
        # Linear layer to bottleneck (Latent Vector)
        latent_vector = self.fc_encode(x)
        
        # --- DECODER FORWARD PASS ---
        x = self.fc_decode(latent_vector)
        
        # Reshape back to 3D volume: [B, C*H*W] -> [B, C, H, W]
        x = x.view(batch_size, self.final_conv_channels, self.final_h, self.final_w)
        
        # Transposed Convolution layers
        reconstructed_spectrogram = self.decoder(x)
        
        # Ensure final output size matches the original input size [1, N_MELS, INPUT_TIME_FRAMES]
        # NOTE: You may need to manually adjust the output_padding in the last ConvTranspose2d layer
        # if your spatial sizes are not perfectly divisible, or you can use F.interpolate().
        reconstructed_spectrogram = reconstructed_spectrogram[:, :, :self.n_mels, :self.time_frames]
        
        return reconstructed_spectrogram, latent_vector

# --- Model Initialization Example ---
# model = AudioAutoencoder(N_MELS, INPUT_TIME_FRAMES, LATENT_DIM)
# print(model)

# DUMMY_INPUT = torch.randn(BATCH_SIZE, 1, N_MELS, INPUT_TIME_FRAMES) 
# output, latent = model(DUMMY_INPUT)
# print(f"Input Shape: {DUMMY_INPUT.shape}")
# print(f"Output Shape: {output.shape}")
# print(f"Latent Vector Shape: {latent.shape}")



# --- EXPORT THE ENCODER ---

def export_encoder(model, latent_dim):
    # Create an instance of the model with the trained weights loaded
    encoder_model = model.cpu().eval() 
    
    # Create a dummy input that matches the expected spectrogram shape (Batch=1 for inference)
    # The latent_vector is the second output of the forward pass
    dummy_input = torch.randn(1, 1, N_MELS, INPUT_TIME_FRAMES)
    
    # Define which axes are dynamic (helpful for future resizing)
    dynamic_axes = {'input': {0: 'batch_size'}, 'latent_vector': {0: 'batch_size'}}
    
    # The function to trace for the Encoder: only the first part of forward()
    def encoder_forward(x):
        x = encoder_model.encoder(x)
        x = x.view(x.size(0), -1)
        latent_vector = encoder_model.fc_encode(x)
        return latent_vector
        
    torch.onnx.export(
        encoder_forward,                  # Model function
        dummy_input,                      # Sample input
        "encoder.onnx",                   # Output file name
        verbose=False,
        opset_version=14,                 # A RECENT, STABLE ONNX OPSET VERSION
        input_names=['spectrogram_input'],
        output_names=['latent_vector'],
        dynamic_axes=dynamic_axes
    )
    print("Encoder exported to encoder.onnx")

# --- Export the Encoder example ---
# export_encoder(model, LATENT_DIM)