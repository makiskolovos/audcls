import os
import subprocess
from multiprocessing.pool import ThreadPool as Pool
from glob import glob

from conf import DATA_PATH
from shlex import quote, split


def raw_to_wav():
    """
    The raw_to_wav function converts all files in the raw folder to .wav format and saves them in the wav folder.
    It uses ffmpeg to do this, which must be installed on your system for this function to work.


    :return: A list of all wav files in the 'wav' folder
    :doc-author: Trelent
    """
    print(f"Seeking all files in {os.path.join(DATA_PATH, 'raw')}")
    all_songs = glob(os.path.join(DATA_PATH, 'raw', '**', '**.**'), recursive=True)
    print(f"Found {len(all_songs)} files.")

    def convert_to_wav(path):
        dir_name, file_name = os.path.split(path)
        new_path = os.path.join(dir_name.replace("raw", "wav"), '.'.join((os.path.splitext(file_name)[0], 'wav')))
        new_dir_name, new_file_name = os.path.split(new_path)
        os.makedirs(new_dir_name, exist_ok=True)
        subprocess.call(split(f'ffmpeg -v 0 -y -i '
                              f'{os.path.join(dir_name, quote(file_name))} '
                              f'{os.path.join(new_dir_name, quote(new_file_name))}'))

    print(f"Converting all files found to .wav and save to {os.path.join(DATA_PATH, 'wav')}")
    with Pool(len(os.sched_getaffinity(0))) as p:
        p.map(convert_to_wav, all_songs)


if __name__ == '__main__':
    raw_to_wav()
