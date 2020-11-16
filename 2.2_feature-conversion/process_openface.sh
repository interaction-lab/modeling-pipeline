#!/bin/bash

# Input & Output Dir
BASE_INPUT_DIR="/media/$(whoami)/M2/2-Processed_Data/Raw_Output_CSVs/Raw_Face2"
BASE_OUTPUT_DIR="/media/chris/M2/2-Processed_Data/Video-OpenFace"

audio_dir="audio"

# {1..3} {5..27}
# For each study
for study_idx in {1..3} {5..27}; do
  echo "Extracting from study: ${study_idx}"
  for position in "left" "right" "center"; do
    echo "Extracting from face: ${position}"
    input_path="${BASE_INPUT_DIR}/${study_idx}/${position}.csv"
    output_path="${BASE_OUTPUT_DIR}/${study_idx}/${position}.csv"
    mkdir -p "${BASE_OUTPUT_DIR}/${study_idx}/"
    python format_openface.py $input_path $output_path
  done
done