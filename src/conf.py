import os

# Main paths
ROOT_PATH = os.path.dirname(__file__).replace('src', '')
INPUTS_PATH = os.path.join(ROOT_PATH, 'inputs')
DATA_PATH = os.path.join(INPUTS_PATH, 'data')

# Raw data parameters
SAMPLE_RATE = 16000
FRAME_LENGTH = 1024
FRAME_STEP = 128
DURATION = 10
N_MELS = 128
NDFTS = round(((SAMPLE_RATE * DURATION) - 1) / (FRAME_LENGTH - (FRAME_LENGTH - FRAME_STEP))) + 1

# Project parameters
PROJECT_NAME = 'mfccs_only'
SAMPLE_TYPE = 'mfccs'
DIRECTORY = os.path.join(ROOT_PATH, 'output')
RESULTS = os.path.join(ROOT_PATH, DIRECTORY, PROJECT_NAME, 'results')
MODEL_DIR = os.path.join(ROOT_PATH, DIRECTORY, PROJECT_NAME, 'model')
HISTORY = os.path.join(RESULTS, 'history.csv')
os.makedirs(RESULTS, exist_ok=True)

# Training parameters
LOSS = 'categorical_crossentropy'
METRICS = ['categorical_accuracy']
OBJECTIVE = 'val_categorical_accuracy'
MAX_TRIALS = 50
EPOCHS = 100
BATCH_SIZE = 128
INPUT_SHAPE = (N_MELS, NDFTS, 3)

# Data parameters
TRAIN_DATA = os.path.join(INPUTS_PATH, 'train.csv')
VAL_DATA = os.path.join(INPUTS_PATH, 'val.csv')
CLASSES = {'dimotiko': 0, 'neo_laiko': 1, 'palio_laiko': 2, 'rempetiko': 3}
