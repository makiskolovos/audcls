import os
from multiprocessing.pool import ThreadPool as Pool
from glob import glob
import cv2
import numpy as np

import librosa
import tensorflow as tf
from conf import DATA_PATH, INPUTS_PATH

SAMPLE_RATE = 16000
FRAME_LENGTH = 1024
FRAME_STEP = 128
DURATION = 10


def load_and_prepare_audio(filename, sample_rate=SAMPLE_RATE, mono=True, frame_length=FRAME_LENGTH,
                           frame_step=FRAME_STEP, n_mels=128, duration=DURATION):
    """
    The load_and_prepare_audio function takes in a filename and returns a tuple of lists
    (spectrogram, chromagram, mfcc) of that audio file. The function first loads the audio
    using librosa's load function. It then splits the audio into 10 second frames (by default).
    For each frame it extracts a mel-scaled spectrogram using librosa's melspectrogram function.
    It then converts this to decibels by taking 10*log10(S) where S is the power spectrum of y
    (the input signal). It then normalizes this between 0-1.

    :param filename: Load the audio file
    :param sample_rate: Set the sample rate of the audio file
    :param mono: Determine if the audio should be converted to mono or not
    :param frame_length: Determine the length of each frame in samples
    :param frame_step: Determine the overlap between frames
    :param n_mels: Specify the number of mel bands to generate
    :param duration: Determine the length of each frame
    :return: A list of spectrograms, chromagrams and mfccs
    :doc-author: Trelent
    """
    audio, _ = librosa.load(filename, sr=sample_rate, mono=mono)
    frames = tf.signal.frame(signal=audio, frame_length=sample_rate * duration,
                             frame_step=sample_rate * duration, pad_end=True)
    spectrograms = []
    chromagrams = []
    mfccs = []
    for frame in frames[1:-1]:  # Ignore first and last 10 seconds.
        # Extract spectrogram using librosa
        spectrogram = librosa.feature.melspectrogram(y=frame.numpy(), sr=sample_rate, n_mels=n_mels, n_fft=frame_length,
                                                     hop_length=frame_step)
        # # Convert spectrogram to decibels
        log_mel_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        # # Normalize the spectrogram between 0-1
        normalized_spectrogram = np.divide(log_mel_spectrogram - np.min(log_mel_spectrogram),  # nans if no freq diffs
                                           np.max(log_mel_spectrogram) - np.min(log_mel_spectrogram))
        normalized_spectrogram = np.nan_to_num(normalized_spectrogram)  # Switch nan to 0
        chroma = librosa.feature.chroma_stft(y=frame.numpy(), sr=sample_rate, n_fft=frame_length, hop_length=frame_step)

        mfcc = librosa.feature.mfcc(y=frame.numpy(), sr=sample_rate, n_fft=frame_length, hop_length=frame_step)
        normalized_mfcc = np.divide(mfcc - np.min(mfcc),  # nans if no freq diffs
                                    np.max(mfcc) - np.min(mfcc))
        spectrograms.append(normalized_spectrogram)
        chromagrams.append(chroma)
        mfccs.append(normalized_mfcc)
    return spectrograms, chromagrams, mfccs


def write_image(wav_path):
    """
    The write_image function takes a wav_path as input and returns nothing.
    It first loads the audio file at wav_path, then it computes its spectrogram, chromagram and mfccs.
    Then for each of these three features (spectrogram, chromagram and mfccs), it saves them in their respective folders
    (spects/, chromas/ or mfccs/) with the same name as the original audio file but with an additional index number to
    distinguish between different instances of this function being called on files that have identical names.

    :param wav_path: Get the path of the wav file
    :return: The path of the image written to disk
    :doc-author: Trelent
    """
    spectrograms, chromas, mfccs = load_and_prepare_audio(wav_path)
    for idx, spect_chromas_mfccs in enumerate(list(zip(spectrograms, chromas, mfccs))):
        spects_path = "_".join((os.path.splitext(wav_path.replace(os.path.join(DATA_PATH, 'wav'),
                                                                  os.path.join(INPUTS_PATH, "spects")))[0], str(idx)))
        chromas_path = "_".join((os.path.splitext(wav_path.replace(os.path.join(DATA_PATH, 'wav'),
                                                                   os.path.join(INPUTS_PATH, "chromas")))[0], str(idx)))
        mfccs_path = "_".join((os.path.splitext(wav_path.replace(os.path.join(DATA_PATH, 'wav'),
                                                                 os.path.join(INPUTS_PATH, "mfccs")))[0], str(idx)))
        os.makedirs(os.path.dirname(spects_path), exist_ok=True)
        os.makedirs(os.path.dirname(chromas_path), exist_ok=True)
        os.makedirs(os.path.dirname(mfccs_path), exist_ok=True)
        np.save(spects_path, spect_chromas_mfccs[0])
        np.save(chromas_path, spect_chromas_mfccs[1])
        np.save(mfccs_path, spect_chromas_mfccs[2])


def write_data():
    """
    The write_data function takes the audio files in the data folder and converts them into images.
    It does this by first creating a directory for each type of image (spectrogram, chromagram, mfcc)
    in the inputs folder. Then it uses multiprocessing to convert all of these files at once.

    :return: The number of files that were processed
    :doc-author: Trelent
    """
    for img_type in ("spects", "chromas", "mfccs"):
        os.makedirs(os.path.join(INPUTS_PATH, f"{img_type}"), exist_ok=True)
    files = glob(os.path.join(DATA_PATH, 'wav', '**', '**.wav'), recursive=True)
    with Pool(len(os.sched_getaffinity(0))) as p:
        p.map(write_image, files)


def save_as_image(paths):
    """
    The save_as_image function takes in a list of paths to the spectrogram, chromagram, and mfccs
    of an audio file. It loads these files into numpy arrays and concatenates them along the last axis
    to create a 3-channel image. The function then saves this image as a .png file in the images folder.

    :param paths: Specify the path of the spectrogram, chromagram and mfccs
    :return: The image
    :doc-author: Trelent
    """

    spect, chroma, mfccs = np.load(paths[0]), np.load(paths[1]), np.load(paths[2])
    image = np.concatenate([
        np.expand_dims(spect, axis=-1),
        np.expand_dims(cv2.resize(chroma, (spect.shape[1], spect.shape[0])), axis=-1),
        np.expand_dims(cv2.resize(mfccs, (spect.shape[1], spect.shape[0])), axis=-1)],
        axis=-1)
    image = image * 255.
    image = cv2.convertScaleAbs(image)
    os.makedirs(os.path.split(paths[0])[0].replace("spects", "images"), exist_ok=True)
    cv2.imwrite(paths[0].replace("spects", "images").replace(".npy", ".png"), image)


def save_images():
    """
    The save_images function takes the .npy files in the spects, chromas, and mfccs folders
    and saves them as images. This is done so that we can use a CNN to classify these images.
    The function uses multiprocessing to speed up this process.

    :return: Nothing
    :doc-author: Trelent
    """
    spect_paths = glob(os.path.join(INPUTS_PATH, "spects/**/**.npy"), recursive=True)
    chroma_paths = list(map(lambda x: x.replace("spects", "chromas"), spect_paths))
    mfccs_paths = list(map(lambda x: x.replace("spects", "mfccs"), spect_paths))
    paths = list(zip(spect_paths, chroma_paths, mfccs_paths))
    with Pool(len(os.sched_getaffinity(0))) as p:
        p.map(save_as_image, paths)


if __name__ == '__main__':
    save_images()
    exit()
