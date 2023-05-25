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
    # Load audio file
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
    for img_type in ("spects", "chromas", "mfccs"):
        os.makedirs(os.path.join(INPUTS_PATH, f"{img_type}"), exist_ok=True)
    files = glob(os.path.join(DATA_PATH, 'wav', '**', '**.wav'), recursive=True)
    with Pool(len(os.sched_getaffinity(0))) as p:
        p.map(write_image, files)


def save_as_image(paths):
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
    spect_paths = glob(os.path.join(INPUTS_PATH, "spects/**/**.npy"), recursive=True)
    chroma_paths = list(map(lambda x: x.replace("spects", "chromas"), spect_paths))
    mfccs_paths = list(map(lambda x: x.replace("spects", "mfccs"), spect_paths))
    paths = list(zip(spect_paths, chroma_paths, mfccs_paths))
    with Pool(len(os.sched_getaffinity(0))) as p:
        p.map(save_as_image, paths)


if __name__ == '__main__':
    save_images()
    exit()
