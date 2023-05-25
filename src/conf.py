import os

ROOT_PATH = os.path.dirname(__file__).replace('src', '')
INPUTS_PATH = os.path.join(ROOT_PATH, 'inputs')
DATA_PATH = os.path.join(INPUTS_PATH, 'data')

PROJECT_NAME = 'mfccs_only'
SAMPLE_TYPE = 'mfccs'
DIRECTORY = os.path.join(ROOT_PATH, 'output')
RESULTS = os.path.join(ROOT_PATH, DIRECTORY, PROJECT_NAME, 'results')
MODEL_DIR = os.path.join(ROOT_PATH, DIRECTORY, PROJECT_NAME, 'model')
HISTORY = os.path.join(RESULTS, 'history.csv')
os.makedirs(RESULTS, exist_ok=True)
TEST = 'test'
LOSS = 'categorical_crossentropy'
METRICS = ['categorical_accuracy']
OBJECTIVE = 'val_categorical_accuracy'
EPOCHS = 5
BATCH_SIZE = 128
INPUT_SHAPE = (128, 1251, 3)

TRAIN_DATA = os.path.join(INPUTS_PATH, 'train.csv')
VAL_DATA = os.path.join(INPUTS_PATH, 'val.csv')
CLASSES = {'dimotiko': 0, 'neo_laiko': 1, 'palio_laiko': 2, 'rempetiko': 3}
