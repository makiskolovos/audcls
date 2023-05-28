import os.path
import pandas as pd

from conf import DIRECTORY
from evaluate_model import Evaluator

chromas_df = pd.read_csv(os.path.join(DIRECTORY, 'chromas_only/results/predictions.csv'))
mfccs_df = pd.read_csv(os.path.join(DIRECTORY, 'mfccs_only/results/predictions.csv'))
spect_df = pd.read_csv(os.path.join(DIRECTORY, 'spect_only/results/predictions.csv'))

new_df = (chromas_df[['0', '1', '2', '3']] + mfccs_df[['0', '1', '2', '3']] + spect_df[['0', '1', '2', '3']]) / 3
evaluate = Evaluator(output=new_df.values)
evaluate.metrics_per_song()
evaluate.metrics_per_sample()
