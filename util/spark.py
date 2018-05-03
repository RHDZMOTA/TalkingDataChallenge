import os
import shutil
import pandas as pd


output_dir = "/home/intersys-rhdzmota/Documents/github/rhdzmota/TalkingDataChallenge/output/submit/20180503-170304-0.9330283721935587-random-forest-submit.csv-folder"


def collect_submit(output_dir, logger):
    logger.info("[function call] collect_submit(output_dir=%s)" % output_dir)
    if not os.path.isdir(output_dir):
        raise ValueError("Variable output_dir=%s is not a dir." % output_dir)

    files = [file for file in os.listdir(output_dir) if (".csv" in file) and (".crc" not in file)]
    csv_file = output_dir.replace("-folder", "")

    df = pd.read_csv(output_dir + "/" + files[0])
    df.to_csv(csv_file, index=None)
    
    for file in files[1:]:
        df = pd.read_csv(output_dir + "/" + file)
        with open(csv_file, "a") as f:
            df.to_csv(f, index=None, header=False)
    shutil.rmtree(output_dir)
    return df
