import torch

# Paths
BASE_PATH = 'data/'
METADATA_PATH = BASE_PATH + 'metadata.csv'
IMAGE_DIR = BASE_PATH + 'images/'

# Model parameters
BATCH_SIZE = 32
NUM_EPOCHS_BASELINE = 30
NUM_EPOCHS_GENERALIST = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training improvements
USE_CLASS_WEIGHTS = True
USE_SCHEDULER = True
USE_EARLY_STOPPING = True
PATIENCE = 5

# Few-shot parameters
K_SHOT = 5
NOVEL_CLASSES = ['MEL', 'SCC']
