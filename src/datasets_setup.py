import os
from glob import glob

import numpy as np
import pandas as pd
from conf import CLASSES, DATA_PATH, INPUTS_PATH
rng = np.random.default_rng(1)
# Get the minimum number of samples presented in all categories
min_song_num = min([len(glob(os.path.join(INPUTS_PATH, 'wav', i, '**'))) for i in CLASSES])

for _class in CLASSES:
    train_list_to_df = []
    val_list_to_df = []

    # Shuffle each class samples
    no_ext_paths = glob(os.path.join(INPUTS_PATH, 'wav', _class, '**'))
    rng.shuffle(no_ext_paths)
    # Select the first min_num_samples for
    no_ext_paths = no_ext_paths[: min_song_num]
    # Derive the first 80% samples as training samples and the rest as validation samples
    train_samples = no_ext_paths[:int(len(no_ext_paths) * 0.8)]
    val_samples = no_ext_paths[int(len(no_ext_paths) * 0.8):]

    # Remove extension .wav from file paths
    train_paths = [os.path.splitext(path)[0] for path in train_samples]
    train_image_paths = [path.replace(os.path.join(DATA_PATH, 'wav'),
                                      os.path.join(INPUTS_PATH, 'images')) for path in train_paths]
    train_image_samples = [glob(path + '**') for path in train_image_paths]
    train_image_samples = [item for sublist in train_image_samples for item in sublist]

    val_paths = [os.path.splitext(path)[0] for path in val_samples]
    val_image_paths = [path.replace(os.path.join(DATA_PATH, 'wav'),
                                    os.path.join(INPUTS_PATH, 'images')) for path in val_paths]
    val_image_samples = [glob(path + '**') for path in val_image_paths]
    val_image_samples = [item for sublist in val_image_samples for item in sublist]

    for train_img_sample in train_image_samples:
        train_list_to_df.append((train_img_sample, CLASSES[_class]))

    for val_img_sample in val_image_samples:
        val_list_to_df.append((val_img_sample, CLASSES[_class]))

    train_df = pd.DataFrame(train_list_to_df, columns=['image', 'class'])
    val_df = pd.DataFrame(val_list_to_df, columns=['image', 'class'])

    train_df.to_csv(os.path.join(INPUTS_PATH, 'train.csv'), mode='a', index=False)
    val_df.to_csv(os.path.join(INPUTS_PATH, 'val.csv'), mode='a', index=False)
