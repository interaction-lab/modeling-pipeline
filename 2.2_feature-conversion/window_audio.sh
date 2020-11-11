#!/bin/bash

# Input & Output Dir
BASE_INPUT_DIR="/media/chris/M2/2-Processed_Data/Audio-Librosa"
BASE_OUTPUT_DIR="/media/chris/M2/2-Processed_Data/Audio-Librosa"

config="./config/windowing_librosa.yml"

# {1..3} {5..27}
# For each study
for study_idx in {1..3} {5..27}; do
  echo "Windowing study: ${study_idx}"
  for position in "audio_features"; do
    input_path="${BASE_INPUT_DIR}/${study_idx}/${position}.csv"
    output_path="${BASE_OUTPUT_DIR}_win/${study_idx}/${position}.csv"
    mkdir -p "${BASE_OUTPUT_DIR}_win/${study_idx}/"
    python windowing.py $input_path $output_path $config
  done
done