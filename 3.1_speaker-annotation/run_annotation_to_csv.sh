#!/bin/bash

# Note: disregard 4 (not recorded), 18, 19 (overlaps), 14 (time is off)

# Progress tracking:
# Completed {1..3} {5..13} {15..17} {20..27}
# annotations_to_csv 14 {18..19}    --> turns.csv

# To fix: 

# Input & Output Dir
BASE_INPUT_DIR="/media/$(whoami)/M2/1-Raw_Data/Videos"
BASE_FEATURES_DIR="/media/$(whoami)/M2/2-Processed_Data"
INPUT="audio/turns.json"
OUTPUT="turns.csv"
FACECSV="right.csv"

for study_idx in {1..3} {5..13} {15..17} {20..27}; do
  echo "Study: ${study_idx}"
  mkdir -p "${BASE_FEATURES_DIR}/Annotations-Turns/${study_idx}"
  python annotations_to_csv.py \
    "${BASE_INPUT_DIR}/${study_idx}/${INPUT}" \
    "${BASE_FEATURES_DIR}/Annotation-Turns/${study_idx}/${OUTPUT}" \
    "${BASE_FEATURES_DIR}/Video-OpenFace/${study_idx}/${FACECSV}"
done