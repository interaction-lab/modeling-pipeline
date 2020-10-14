#!/bin/bash

# Input & Output Dir
BASE_INPUT_DIR="/media/$(whoami)/M2/1-Raw_Data/Videos"
BASE_OUTPUT_DIR="/media/chris/M2/2-Processed_Data/Raw_Audio-openSMILE"
OPENSMILE_DIR="/home/chris/Programs/opensmile-2.3.0"
audio_dir="audio"

# For each study
for study_idx in {1..3} {5..27}; do
  echo "Extracting from study: ${study_idx}"
  audio_path="${BASE_INPUT_DIR}/${study_idx}/${audio_dir}/audio.wav"
  csv_path="${BASE_OUTPUT_DIR}/${study_idx}"
  mkdir -p "${BASE_OUTPUT_DIR}/${study_idx}/"
  python openSMILE_features.py $audio_path $csv_path $OPENSMILE_DIR
done


# Librosa is 36961 long
# SMILE
# frameStep = 0.0333333333 for 36957