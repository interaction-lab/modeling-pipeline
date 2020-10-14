#!/bin/bash

# Input & Output Dir
BASE_INPUT_DIR="/media/$(whoami)/M2/3- Processed Features/Audio"
BASE_OUTPUT_DIR="/media/$(whoami)/M2/3- Processed Features/Organized Audio"

output_dir="${BASE_OUTPUT_DIR}"

# For each study
for study_idx in {1..3} {5..27}; do
  echo "Moving data for: ${study_idx}"
  input_dir="${BASE_INPUT_DIR}/$study_idx"

  # For the relevant files
  for filename in "pitch" "power"; do
    echo "processing \"${input_dir}/${filename}.csv\""
    cp "${input_dir}/${filename}.csv" "${output_dir}/study_${study_idx}_${filename}.csv"
  done
  
done
