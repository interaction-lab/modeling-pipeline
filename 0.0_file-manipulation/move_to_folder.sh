#!/bin/bash

# Input & Output Dir
BASE_INPUT_DIR="/media/$(whoami)/M2/1-Raw_Data/Videos"

output_dir="audio"

# For each study
for study_idx in {1..3} {5..27}; do
  echo "Moving data for: ${study_idx}"
  input_dir="${BASE_INPUT_DIR}/$study_idx"

  # For the relevant files
  for filename in "audio.flac" "transcript.flac" "transcript.mp3"; do
    echo "moving \"${input_dir}/${filename}\""
    mkdir --parents "${input_dir}/$output_dir"
    mv "${input_dir}/${filename}" "${input_dir}/$output_dir/${filename}"
  done
  
done
