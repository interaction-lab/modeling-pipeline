#!/bin/bash

# Input & Output Dir
BASE_INPUT_DIR="/media/$(whoami)/M2/1-Raw_Data/Videos"
BASE_OUTPUT_DIR="/media/chris/M2/2-Processed_Data/Raw_Audio-Librosa"

audio_dir="audio"

# For each study
for study_idx in 18; do
  echo "Extracting from study: ${study_idx}"
  audio_path="${BASE_INPUT_DIR}/${study_idx}/${audio_dir}/audio.wav"
  csv_path="${BASE_OUTPUT_DIR}/${study_idx}/audio_features.csv"
  mkdir -p "${BASE_OUTPUT_DIR}/${study_idx}/"
  python librosa_features.py $audio_path $csv_path
done
