import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

DATA_FOLDER = os.environ.get("DATA_FOLDER")
CONF_FOLDER = os.environ.get("CONF_FOLDER")
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER")
SUBMIT_FOLDER = os.environ.get("SUBMIT_FOLDER")
TRAIN_FOLDER = os.environ.get("TRAIN_FOLDER")
SUBMIT_CSV = os.environ.get("SUBMIT_CSV")
RESULTS_CSV = os.environ.get("RESULTS_CSV")
DROPBOX_DATA_URL = os.environ.get("DROPBOX_DATA_URL")
TRAIN_CSV = os.environ.get("TRAIN_CSV")
TRAIN_SAMPLE_CSV = os.environ.get("TRAIN_SAMPLE_CSV")
TEST_CSV = os.environ.get("TEST_CSV")
LOG_FOLDER = os.environ.get("LOG_FOLDER")

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(PROJECT_DIR, DATA_FOLDER)
OUTPUT_PATH = os.path.join(PROJECT_DIR, OUTPUT_FOLDER)
SUBMIT_PATH = os.path.join(OUTPUT_PATH, SUBMIT_FOLDER)
TRAIN_PATH = os.path.join(OUTPUT_PATH, TRAIN_FOLDER)

class FilesConfig:

    class Paths:
        data = DATA_PATH
        output = OUTPUT_PATH
        conf = os.path.join(PROJECT_DIR, CONF_FOLDER)
        logs = os.path.join(PROJECT_DIR, LOG_FOLDER)
        submit = SUBMIT_PATH
        train = TRAIN_PATH

    class Names:
        train_data = os.path.join(DATA_PATH, TRAIN_CSV)
        train_sample_data = os.path.join(DATA_PATH, TRAIN_SAMPLE_CSV)
        submit_data = os.path.join(DATA_PATH, TEST_CSV)
        submit_output = os.path.join(SUBMIT_PATH, SUBMIT_CSV)
        results = os.path.join(TRAIN_PATH, RESULTS_CSV)

