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
        input1, input2, input3 = None, None, None
        pretrained_models = {'efficientnet_v2b3': efficientnet_v2.EfficientNetV2B3,
                             'efficientnet_b7': efficientnet.EfficientNetB7,
                             'inception_resnet_v2': inception_resnet_v2.InceptionResNetV2,
                             'inception_v3': inception_v3.InceptionV3
                             }

        preprocess = {'efficientnet_v2b3': efficientnet_v2.preprocess_input,
                      'efficientnet_b7': efficientnet.preprocess_input,
                      'inception_resnet_v2': inception_resnet_v2.preprocess_input,
                      'inception_v3': inception_v3.preprocess_input
                      }

        dense_nodes = hp.Int('dense_nodes', min_value=128, max_value=4096, step=2, sampling='log')
        do_rate = hp.Float('dropout', min_value=0.0, max_value=0.7, step=0.01)

        if SAMPLE_TYPE == 'all':
            pretrained1 = pretrained_models['efficientnet_v2b3'](include_top=False, input_shape=INPUT_SHAPE)
            pretrained2 = pretrained_models['efficientnet_v2b3'](include_top=False, input_shape=INPUT_SHAPE)
            pretrained3 = pretrained_models['efficientnet_v2b3'](include_top=False, input_shape=INPUT_SHAPE)
            pretrained1._name = 'spects_pre'
            pretrained2._name = 'mfccs_pre'
            pretrained3._name = 'chromas_pre'
            pretrained1.trainable = False
            pretrained2.trainable = False
            pretrained3.trainable = False
            input1 = keras.layers.Input(shape=INPUT_SHAPE, name='spects')
            input2 = keras.layers.Input(shape=INPUT_SHAPE, name='mfccs')
            input3 = keras.layers.Input(shape=INPUT_SHAPE, name='chromas')
            preproc1 = tf.keras.layers.Lambda(preprocess['efficientnet_v2b3'], name='preproc_spects')(input1)
            preproc2 = tf.keras.layers.Lambda(preprocess['efficientnet_v2b3'], name='preproc_mfccs')(input2)
            preproc3 = tf.keras.layers.Lambda(preprocess['efficientnet_v2b3'], name='preproc_chromas')(input3)
            pre_out1 = pretrained1(preproc1)
            pre_out2 = pretrained2(preproc2)
            pre_out3 = pretrained3(preproc3)
            pre_out = keras.layers.Average()([pre_out1, pre_out2, pre_out3])
        else:
            pretrained_model_name = hp.Choice('pretrained_model', list(pretrained_models.keys()))
            pretrained = pretrained_models[pretrained_model_name](include_top=False, input_shape=INPUT_SHAPE)
            pretrained.trainable = False
            input1 = keras.layers.Input(shape=INPUT_SHAPE, name='inputs')
            preproc1 = tf.keras.layers.Lambda(preprocess[pretrained_model_name])(input1)
            pre_out = pretrained(preproc1)
        flatten = keras.layers.Flatten()(pre_out)
        dense = keras.layers.Dense(units=dense_nodes, activation='relu')(flatten)
        dense = keras.layers.BatchNormalization()(dense)
        dropout = keras.layers.Dropout(rate=do_rate)(dense)
        output = keras.layers.Dense(len(CLASSES), activation='softmax', name='class')(dropout)

        if SAMPLE_TYPE == 'all':
            model = keras.models.Model([input1, input2, input3], output)
        else:
            model = keras.models.Model(input1, output)

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
