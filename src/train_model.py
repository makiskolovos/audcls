import os

import tensorflow as tf
import keras_tuner as kt
import pandas as pd

from model import CustomHyperModel
from evaluate_model import Evaluator
from data_loader import DataLoader
from conf import DIRECTORY, OBJECTIVE, PROJECT_NAME, MODEL_DIR, HISTORY, SAMPLE_TYPE, EPOCHS, BATCH_SIZE, MAX_TRIALS


def hp_optimization():
    """
    The hp_optimization function is a wrapper for the KerasTuner API. It takes in no arguments, and returns a compiled
    model with optimized hyperparameters. The function uses Bayesian Optimization to search through the space of
    hyperparameters defined by CustomHyperModel(). The objective function used to evaluate each model is defined by
    OBJECTIVE, which calls train_and_evaluate() on each model.

    :return: The best model
    :doc-author: Trelent
    """
    tuner = kt.BayesianOptimization(objective=OBJECTIVE, directory=DIRECTORY, project_name=PROJECT_NAME,
                                    hypermodel=CustomHyperModel(), max_trials=MAX_TRIALS, overwrite=False, seed=None)

    tuner.search()
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    [print(f"{name}: {best_hyperparameters.get(name)}") for name in best_hyperparameters.values]

    best_model = tuner.hypermodel.build(best_hyperparameters)

    return best_model


def train_model(model):
    """
    The train_model function trains the model.

    :param model: Specify the model to be trained
    :return: The trained model
    :doc-author: Trelent
    """
    if not os.path.exists(MODEL_DIR):
        es_cb = tf.keras.callbacks.EarlyStopping(monitor=OBJECTIVE,
                                                 start_from_epoch=5,
                                                 patience=10,
                                                 restore_best_weights=True)

        model.fit(x=DataLoader(mode='train', sample_type=SAMPLE_TYPE).load_ds(batch_size=BATCH_SIZE),
                  validation_data=DataLoader(mode='eval', sample_type=SAMPLE_TYPE).load_ds(batch_size=BATCH_SIZE),
                  callbacks=[es_cb],
                  epochs=EPOCHS,
                  verbose=1)

        pd.DataFrame(model.history.history).to_csv(HISTORY, index=False)
        model.save(MODEL_DIR)
        return model


def evaluate_model(trained_model):
    """
    The evaluate_model function takes a trained model and evaluates it on the test set.
    It saves metrics per sample, metrics per song, and writes results to a csv file.

    :param trained_model: Pass the trained model to the evaluator class
    :return: None
    :doc-author: Trelent
    """
    evaluator = Evaluator(trained_model)
    evaluator.metrics_per_sample()
    evaluator.metrics_per_song()
    evaluator.results_to_csv()


if __name__ == '__main__':
    evaluate_model(train_model(hp_optimization()))
    exit()
