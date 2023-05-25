import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from conf import CLASSES, MODEL_DIR, RESULTS, SAMPLE_TYPE
from data_loader import DataLoader


class Evaluator:
    def __init__(self):
        """
        It loads a model from disk (using tf.keras), and then uses that model to predict on a dataset.

        :return: The results_df
        :doc-author: Trelent
        """
        self.cols = ['image', '0', '1', '2', '3', 'predicted', 'target']

        self.dl = DataLoader(mode='predict', sample_type=SAMPLE_TYPE)
        self.ds = self.dl.load_ds(batch_size=200)
        self.df = self.dl.load_df()

        self.model = tf.keras.models.load_model(MODEL_DIR, compile=True)
        self.output = self.model.predict(self.ds)
        self.predicted_class = np.argmax(self.output, axis=-1)  # Get predicted class
        self.targets = self.df['class']

        results_dict = {str(i): self.output[:, i] for i in range(len(CLASSES))}
        results_dict['predicted'] = self.predicted_class
        results_dict['image'] = self.df['image'].map(lambda x: os.path.splitext(os.path.split(x)[1])[0])
        results_dict['target'] = self.df['class']
        self.results_df = pd.DataFrame(results_dict)
        self.results_df = self.results_df[self.cols]

    def results_to_csv(self):
        """
        The results_to_csv function takes the results dataframe and saves it to a csv file in the RESULTS directory.
        The index is set to False so that no row numbers are saved.

        :return: A csv file with the predictions
        :doc-author: Trelent
        """
        self.results_df.to_csv(os.path.join(RESULTS, 'predictions.csv'), index=False)

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
        self.calculate_metrics(y_true=self.targets, y_pred=self.predicted_class, file_name='per_sample')

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
        self.results_df['song'] = self.results_df['image'].map(lambda x: x.split('_')[0])
        songs = self.results_df['song'].unique()
        y_pred = []
        y_true = []
        for song in songs:
            group = self.results_df.loc[self.results_df['song'] == song][['0', '1', '2', '3', 'target']].reindex()
            y_pred.append(np.argmax(group[['0', '1', '2', '3']].mean().values))
            y_true.append(int(group['target'].mean()))
        self.calculate_metrics(y_true=y_true, y_pred=y_pred, file_name='per_song')

    @staticmethod
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
        with open(os.path.join(RESULTS, f'{file_name}_report'), "w") as f:
            f.write(classification_report(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3],
                                          target_names=CLASSES.keys(), digits=3))
        cm = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, normalize='true',
                                                     values_format='.2%', display_labels=CLASSES.keys())
        cm.figure_.savefig(os.path.join(RESULTS, f'{file_name}_cm_norm.png'))
        cm2 = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, normalize=None,
                                                      values_format=None, display_labels=CLASSES.keys())
        cm2.figure_.savefig(os.path.join(RESULTS, f'{file_name}_cm_raw.png'))


if __name__ == '__main__':
    pass
