import tensorflow as tf
from keras.applications import efficientnet_v2, efficientnet, inception_resnet_v2, inception_v3
from tensorflow import keras
from conf import CLASSES, BATCH_SIZE, OBJECTIVE, INPUT_SHAPE, SAMPLE_TYPE, EPOCHS, LOSS, METRICS
import keras_tuner as kt

from data_loader import DataLoader


class CustomHyperModel(kt.HyperModel):

    def build(self, hp):
        """
        The build function is where you define the model architecture.
        The function takes a single argument, hp, which is an instance of HyperParameters.
        You can use this object to specify hyperparameters for your model and then sample from them:

        :param self: Represent the instance of the class
        :param hp: Access the hyperparameters
        :return: A compiled keras model
        :doc-author: Trelent
        """
        pretrained_models = {
            'efficientnet_v2b3': efficientnet_v2.EfficientNetV2B3,
            'efficientnet_b7': efficientnet.EfficientNetB7,
            'inception_resnet_v2': inception_resnet_v2.InceptionResNetV2,
            'inception_v3': inception_v3.InceptionV3
        }

        preprocess = {'efficientnet_v2b3': efficientnet_v2.preprocess_input,
                      'efficientnet_b7': efficientnet.preprocess_input,
                      'inception_resnet_v2': inception_resnet_v2.preprocess_input,
                      'inception_v3': inception_v3.preprocess_input
                      }

        pretrained_model_name = hp.Choice('pretrained_model', list(pretrained_models.keys()))  # efficientnet_v2b3

        dense_1_nodes = hp.Int('dense_1_nodes', min_value=256, max_value=2048, step=2, sampling='log')
        do_rate = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.01)
        pretrained = pretrained_models[pretrained_model_name](include_top=False, input_shape=INPUT_SHAPE)
        pretrained.trainable = False

        image_input = keras.layers.Input(shape=INPUT_SHAPE, name='inputs')
        preproc = tf.keras.layers.Lambda(preprocess[pretrained_model_name])(image_input)
        pretrained = pretrained(preproc)

        flatten = keras.layers.Flatten()(pretrained)
        dense_1 = keras.layers.Dense(units=dense_1_nodes, activation="relu")(flatten)
        dense_1 = keras.layers.BatchNormalization()(dense_1)
        dropout = keras.layers.Dropout(rate=do_rate)(dense_1)
        output = keras.layers.Dense(len(CLASSES), activation='softmax', name='class')(dropout)
        model = keras.models.Model(image_input, output)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(optimizer=opt, loss=LOSS, metrics=METRICS)
        return model

    def fit(self, hp, model, *args, **kwargs):
        """
        The fit function is the main function of this class. It takes in a model, and trains it on the data provided by
        the DataLoader class. The fit function also uses an early stopping callback to prevent overfitting.

        :param hp: Pass the hyperparameters to be used for training
        :param model: Pass the model to be trained
        :return: The history of the training
        :doc-author: Trelent
        """
        train_ds = DataLoader(mode='train', sample_type=SAMPLE_TYPE).load_ds(batch_size=BATCH_SIZE)
        val_ds = DataLoader(mode='eval', sample_type=SAMPLE_TYPE).load_ds(batch_size=BATCH_SIZE)
        es_cb = tf.keras.callbacks.EarlyStopping(start_from_epoch=5, patience=10,
                                                 monitor=OBJECTIVE, verbose=1)
        return model.fit(x=train_ds, validation_data=val_ds, callbacks=[es_cb],
                         epochs=EPOCHS, verbose=1)


if __name__ == '__main__':
    exit()
