source venv/bin/activate
python setup.py
kaggle competitions download -c talkingdata-adtracking-fraud-detection -f train.csv.zip --force
mv ~/.kaggle/competitions/talkingdata-adtracking-fraud-detection/train.csv.zip data/
upzip data/train.csv
mv data/mnt/ssd/kaggle-talkingdata2/competition_files/train.csv data/
rm -rf data/mnt

