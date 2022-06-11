
"""## Configurations"""
RUN_DATA_INITIALIZATION = True
RUN_MODEL_TRAINING = True
ONE_HOT_ENCODE = True
TUNE_HYPERPARAMETERS = False

"""File Path Constants"""
DATASET_PATH = 'data/'
PROCESSED_PATH = DATASET_PATH + 'processed_files.txt'
HDF5_PATH = DATASET_PATH + 'data.hdf5'
CHECKPOINT_PATH = 'checkpoints/'
MODEL_PATH = 'models/segmentation/'

DATASET_SIZE = 1200

# 60:20:20 Split
train_end = (DATASET_SIZE * 3) // 5
valid_end = (DATASET_SIZE // 5) + train_end

"""Model Parameters"""
n_filters = 32
n_classes = 2 if ONE_HOT_ENCODE else 1
height = 104
width = 88
image_channels = 3
