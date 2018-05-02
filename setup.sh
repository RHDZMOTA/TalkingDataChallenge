#!/usr/bin/env bash

echo "Running setup.py..."
source venv/bin/activate
python setup.py
deactivate

echo "Downloading kaggle train dataset..."
kaggle competitions download -c talkingdata-adtracking-fraud-detection -f train.csv.zip --force
mv ~/.kaggle/competitions/talkingdata-adtracking-fraud-detection/train.csv.zip data/

echo "Unzip and move train file."
unzip data/train.csv
mv mnt/ssd/kaggle-talkingdata2/competition_files/train.csv data/
rm -rf mnt
