import os

from util.timeformat import now
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

DATA_FOLDER = os.environ.get("DATA_FOLDER")
CONF_FOLDER = os.environ.get("CONF_FOLDER")
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER")
SUBMIT_FOLDER = os.environ.get("SUBMIT_FOLDER")
TRAIN_FOLDER = os.environ.get("TRAIN_FOLDER")
LOG_FILE = os.environ.get("LOG_FILE")
SUBMIT_CSV = os.environ.get("SUBMIT_CSV")
RESULTS_CSV = os.environ.get("RESULTS_CSV")
DROPBOX_DATA_URL = os.environ.get("DROPBOX_DATA_URL")
TRAIN_CSV = os.environ.get("TRAIN_CSV")
TRAIN_SAMPLE_CSV = os.environ.get("TRAIN_SAMPLE_CSV")
TEST_CSV = os.environ.get("TEST_CSV")
LOG_FOLDER = os.environ.get("LOG_FOLDER")
TRAIN_LARGE_FILE_PERCENTAGE = float(os.environ.get("TRAIN_LARGE_FILE_PERCENTAGE"))

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(PROJECT_DIR, DATA_FOLDER)
OUTPUT_PATH = os.path.join(PROJECT_DIR, OUTPUT_FOLDER)
SUBMIT_PATH = os.path.join(OUTPUT_PATH, SUBMIT_FOLDER)
TRAIN_PATH = os.path.join(OUTPUT_PATH, TRAIN_FOLDER)
LOG_PATH = os.path.join(PROJECT_DIR, LOG_FOLDER)


class FilesConfig:

    class Paths:
        conf = os.path.join(PROJECT_DIR, CONF_FOLDER)
        submit = SUBMIT_PATH
        output = OUTPUT_PATH
        train = TRAIN_PATH
        data = DATA_PATH
        logs = LOG_PATH

    class Names:
        train_sample_data = os.path.join(DATA_PATH, TRAIN_SAMPLE_CSV)
        submit_output = os.path.join(SUBMIT_PATH, SUBMIT_CSV)
        train_data = os.path.join(DATA_PATH, TRAIN_CSV)
        submit_data = os.path.join(DATA_PATH, TEST_CSV)
        results = os.path.join(TRAIN_PATH, RESULTS_CSV)
        logger = os.path.join(LOG_PATH, LOG_FILE)


class LogConf:
    path = FilesConfig.Names.logger.format(date=now())
    format = '%(asctime)s %(levelname)s:%(message)s'
    datefmt = '%m/%d/%Y %I:%M:%S %p'

    @staticmethod
    def create(logging):
        logging.basicConfig(format=LogConf.format, filename=LogConf.path, datefmt=LogConf.datefmt, level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler())
        return logger
