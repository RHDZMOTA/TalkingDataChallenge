import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

DATA_FOLDER = os.environ.get("DATA_FOLDER")
CONF_FOLDER = os.environ.get("CONF_FOLDER")
DROPBOX_DATA_URL = os.environ.get("DROPBOX_DATA_URL")
TRAIN_CSV = os.environ.get("TRAIN_CSV")
TRAIN_SAMPLE_CSV = os.environ.get("TRAIN_SAMPLE_CSV")
TEST_CSV = os.environ.get("TEST_CSV")
LOG_FOLDER = os.environ.get("LOG_FOLDER")

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(PROJECT_DIR, DATA_FOLDER)


class FilesConfig:

    class Paths:
        data = DATA_PATH
        conf = os.path.join(PROJECT_DIR, CONF_FOLDER)
        logs = os.path.join(PROJECT_DIR, LOG_FOLDER)

    class Names:
        train_data = os.path.join(DATA_PATH, TRAIN_CSV)
        train_sample_data = os.path.join(DATA_PATH, TRAIN_SAMPLE_CSV)
        submit_data = os.path.join(DATA_PATH, TEST_CSV)
