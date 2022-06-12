TRAIN_MODEL = False
TUNE_HYPERPARAMETERS = False

BASE_PATH = '../'
DATASET_PATH = 'data/'
HDF5_PATH = BASE_PATH + DATASET_PATH
CHECKPOINT_PATH = BASE_PATH + 'checkpoints'
MODEL_PATH = BASE_PATH + 'models/inpainting_unet/'

DATASET_SIZE = 1000

# 90:10 Train-Test Split
test_start = (DATASET_SIZE * 9) // 10

# Model Hyperparmeters
height = 104
width = 88
n_channels = 3
n_filters = 32