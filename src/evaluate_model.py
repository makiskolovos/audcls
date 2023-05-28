import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from conf import CLASSES, RESULTS_DIR, SAMPLE_TYPE, BATCH_SIZE
from data_loader import DataLoader


def calculate_metrics(y_true, y_pred, file_name):
    """
    The calculate_metrics function takes in the true labels and predicted labels of a model,
    and outputs a classification report and two confusion matrices. The first confusion matrix is normalized,
    while the second one is not. Both are saved as png files.

    :param y_true: Pass the true labels of the test set
    :param y_pred: Calculate the metrics
    :param file_name: Save the results to a file
    :return: A classification report and two confusion matrices
    :doc-author: Trelent
    """
    with open(os.path.join(RESULTS_DIR, f'{file_name}_report'), "w") as f:
        f.write(classification_report(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3],
                                      target_names=CLASSES.keys(), digits=3))
    cm = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, normalize='true',
                                                 values_format='.2%', display_labels=CLASSES.keys())
    cm.figure_.savefig(os.path.join(RESULTS_DIR, f'{file_name}_cm_norm.png'))
    cm2 = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, normalize=None,
                                                  values_format=None, display_labels=CLASSES.keys())
    cm2.figure_.savefig(os.path.join(RESULTS_DIR, f'{file_name}_cm_raw.png'))


class Evaluator:
    def __init__(self, model=None, output=None):
        """
        It loads a model from disk (using tf.keras), and then uses that model to predict on a dataset.

        :return: The results_df
        :doc-author: Trelent
        """
        self.cols = ['song', 'sample', '0', '1', '2', '3', 'y_pred', 'y_true']

        self.dl = DataLoader(mode='predict', sample_type=SAMPLE_TYPE)
        self.ds = self.dl.load_ds(batch_size=BATCH_SIZE)
        self.df = self.dl.load_df()
        if output is not None:
            self.output = output
        else:
            self.output = model.predict(self.ds)
        self.y_pred = np.argmax(self.output, axis=-1)  # Get predicted class
        results_dict = {str(i): self.output[:, i] for i in range(len(CLASSES))}

        self.y_true = self.df['class']
        results_dict['sample'] = self.df['image'].map(lambda x: os.path.splitext(os.path.split(x)[1])[0])
        results_dict['song'] = results_dict['sample'].map(lambda x: x.split('_')[0])

        results_dict['y_true'] = self.df['class']
        results_dict['y_pred'] = self.y_pred

        self.results_df = pd.DataFrame(results_dict)
        self.results_df = self.results_df[self.cols]

    def results_to_csv(self):
        """
        The results_to_csv function takes the results dataframe and saves it to a csv file in the RESULTS directory.
        The index is set to False so that no row numbers are saved.

        :return: A csv file with the predictions
        :doc-author: Trelent
        """
        self.results_df.to_csv(os.path.join(RESULTS_DIR, 'predictions.csv'), index=False)

    def metrics_per_sample(self):
        """
        The metrics_per_sample function calculates the following metrics for each sample in the dataset:
            - Accuracy
            - Precision
            - Recall (Sensitivity)
            - Specificity (True Negative Rate)

        :param self: Represent the instance of the class
        :return: The precision, recall, and f-measure for each sample
        :doc-author: Trelent
        """
        calculate_metrics(y_true=self.y_true, y_pred=self.y_pred, file_name='per_sample')

    def metrics_per_song(self):
        """
        The metrics_per_song function takes the results_df dataframe and creates a new column called 'song' which is
        populated by the first part of each image name (i.e. song name). It then iterates through all unique songs,
        creating a group for each song which contains only rows with that particular song's images in it.
        The function then takes the mean of each column in this group, and appends these means to y_pred and y_true
        lists as appropriate.

        :param self: Refer to the current object
        :return: The metrics for the model per song
        :doc-author: Trelent
        """
        songs = self.results_df['song'].unique()
        y_pred = []
        y_true = []
        for song in songs:
            group = self.results_df.loc[self.results_df['song'] == song][self.cols[2:]].reindex()
            y_pred.append(int(group[['0', '1', '2', '3']].mean().idxmax()))
            y_true.append(group['y_true'].mean())
        calculate_metrics(y_true=y_true, y_pred=y_pred, file_name='per_song')


if __name__ == '__main__':
    pass
