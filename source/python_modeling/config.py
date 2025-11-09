# --- AUDIO PARAMETERS ---
SAMPLE_RATE = 22050               # The target sample rate of your audio data
FFT_SIZE = 1024                   # N_FFT: Window size for STFT/Spectrogram
HOP_LENGTH = 256                  # Stride: Distance between STFT frames
N_MELS = 128                      # N_MELS: Number of frequency bins in the Mel Spectrogram
AUDIO_DURATION = 1.0              # Duration of audio clips in seconds

# Calculate the input size (Time Frames) based on duration and hop_length
# Formula: floor( (duration * sr - fft_size) / hop_length ) + 1
# This is used for the Linear layer input/output calculation
INPUT_TIME_FRAMES = int((AUDIO_DURATION * SAMPLE_RATE - FFT_SIZE) // HOP_LENGTH) + 1

# --- MODEL PARAMETERS ---
LATENT_DIM = 64                   # THE SIZE OF THE LATENT VECTOR ("SOUND DNA"). THIS IS THE PRIMARY USER-ADJUSTABLE PARAMETER COUNT.
BATCH_SIZE = 32                   # Batch size for training
LEARNING_RATE = 1e-4              # Optimizer learning rate
NUM_EPOCHS = 50                   # Number of training epochs