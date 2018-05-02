from conf.settings import FilesConfig
from util.download import download_file

import os
import subprocess

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def download_raw_data():
    download_file("train_sample.csv", "/TalkingData", FilesConfig.Names.train_sample_data)
    download_file("test.csv", "/TalkingData", FilesConfig.Names.train_sample_data)

def create_dirs():
    create_dir(FilesConfig.Paths.data)
    create_dir(FilesConfig.Paths.logs)


if __name__ == "__main__":
    create_dirs()
    download_raw_data()
