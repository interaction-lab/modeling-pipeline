#!/bin/bash

# Input & Output Dir
BASE_INPUT_DIR="/media/$(whoami)/M2/1-Raw_Data/Videos"

audio_dir="audio"

# For each study
for study_idx in {1..3} {5..27}; do
  echo "Study: ${study_idx}"
  audio_file="${BASE_INPUT_DIR}/${study_idx}/${audio_dir}/audio.flac"
  copy="${BASE_INPUT_DIR}/${study_idx}/${audio_dir}/audio.wav"
  ffmpeg -i $audio_file $copy
  
done
