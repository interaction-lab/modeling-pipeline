#!/bin/bash

# Note: disregard 4 (not recorded), 18, 19 (overlaps), 14 (time is off)

# Progress tracking:
# 0 label_speakers 14 {18..19}    --> labels.json
# 1 check_empty_spaces 14 {18..19}--> labels.json
# 2 fix_speakers 14 {18..19}      --> labels_fixed.json
# 3 merge_speakers 14 {18..19}    --> turns.json
# 4 review_labels {1..3} {5..13} {15..17} {20..27}

# Input & Output Dir
BASE_INPUT_DIR="/media/$(whoami)/M2/1-Raw_Data/Videos"
INPUT="turns.json"
OUTPUT="reviewed_turns.json"
STEP="4"

for study_idx in 1; do
  echo "Study: ${study_idx}"
  python annotate_utterances.py \
    "${BASE_INPUT_DIR}/${study_idx}/group${study_idx}.mp4" \
    "${BASE_INPUT_DIR}/${study_idx}/audio/${INPUT}" \
    "${BASE_INPUT_DIR}/${study_idx}/audio/${OUTPUT}" \
    $STEP
done