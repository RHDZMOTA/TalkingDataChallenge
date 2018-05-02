import logging
import os

from conf.settings import FilesConfig, LogConf
from util.download import download_file


def create_dir(path):
    logger.info("[function call] create_dir(path=%s)", path)
    if not os.path.exists(path):
        os.mkdir(path)


def download_raw_data():
    download_file("train_sample.csv", "/TalkingData", FilesConfig.Names.train_sample_data, logger)
    download_file("test.csv", "/TalkingData", FilesConfig.Names.submit_data, logger)


def create_dirs():
    create_dir(FilesConfig.Paths.data)
    create_dir(FilesConfig.Paths.logs)
    create_dir(FilesConfig.Paths.output)
    create_dir(FilesConfig.Paths.submit)
    create_dir(FilesConfig.Paths.train)


if __name__ == "__main__":
    logger = LogConf.create(logging)
    create_dirs()
    download_raw_data()
