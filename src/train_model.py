import os

import tensorflow as tf
import keras_tuner as kt
import pandas as pd

from model import CustomHyperModel
from evaluate_model import Evaluator
from data_loader import DataLoader
from conf import DIRECTORY, OBJECTIVE, PROJECT_NAME, MODEL_DIR, HISTORY, SAMPLE_TYPE, EPOCHS, BATCH_SIZE


def hp_optimization():
    """
    The hp_optimization function is a wrapper for the KerasTuner API. It takes in no arguments, and returns a compiled
    model with optimized hyperparameters. The function uses Bayesian Optimization to search through the space of
    hyperparameters defined by CustomHyperModel(). The objective function used to evaluate each model is defined by
    OBJECTIVE(), which calls train_and_evaluate() on each model.

    :return: The best model
    :doc-author: Trelent
    """
    tuner = kt.BayesianOptimization(objective=OBJECTIVE, directory=DIRECTORY, project_name=PROJECT_NAME,
                                    hypermodel=CustomHyperModel(), max_trials=1, overwrite=False, seed=None)

    tuner.search()
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    [print(f"{name}: {best_hyperparameters.get(name)}") for name in best_hyperparameters.values]

    best_model = tuner.hypermodel.build(best_hyperparameters)
    return best_model


model = hp_optimization()

train_ds = DataLoader(mode='train', sample_type=SAMPLE_TYPE).load_ds(batch_size=BATCH_SIZE)
val_ds = DataLoader(mode='eval', sample_type=SAMPLE_TYPE).load_ds(batch_size=BATCH_SIZE)


if not os.path.exists(MODEL_DIR):
    es_cb = tf.keras.callbacks.EarlyStopping(monitor=OBJECTIVE,
                                             start_from_epoch=5, patience=10,
                                             restore_best_weights=True)

    history = model.fit(x=train_ds, validation_data=val_ds, callbacks=[es_cb], epochs=EPOCHS, verbose=1)
    model.save(MODEL_DIR)
    pd.DataFrame(history.history).to_csv(HISTORY)

# Run after saving model
evaluator = Evaluator()
evaluator.metrics_per_sample()
evaluator.metrics_per_song()
evaluator.results_to_csv()

exit()
