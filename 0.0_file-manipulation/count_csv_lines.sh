#!/bin/bash

# Input & Output Dir
BASE_INPUT_DIR="/media/$(whoami)/M2/2-Processed_Data"

output_dir="audio"
# {1..3} {5..27}
# For each study
for study_idx in {1..3} {5..27}; do
  echo ""
  echo "Counting lines for: ${study_idx}"
  echo "Annotation-Turns   " & wc -l "${BASE_INPUT_DIR}/Annotation-Turns/${study_idx}/turns.csv"
  echo "Annotation-Turns_win" & wc -l "${BASE_INPUT_DIR}/Annotation-Turns_win/${study_idx}/turns.csv"
  echo ""
  echo "Video-OpenFace-l_win" & wc -l "${BASE_INPUT_DIR}/Video-OpenFace_win/${study_idx}/left.csv"
  echo "Video-OpenFace-c_win" & wc -l "${BASE_INPUT_DIR}/Video-OpenFace_win/${study_idx}/center.csv"
  echo "Video-OpenFace-r_win" & wc -l "${BASE_INPUT_DIR}/Video-OpenFace_win/${study_idx}/right.csv"
  echo "Audio-Librosa_win" & wc -l "${BASE_INPUT_DIR}/Audio-Librosa_win/${study_idx}/audio_features.csv"
  echo ""
  echo "Audio-Librosa   " & wc -l "${BASE_INPUT_DIR}/Audio-Librosa/${study_idx}/audio_features.csv"
  echo "Video-OpenFace-l   " & wc -l "${BASE_INPUT_DIR}/Video-OpenFace/${study_idx}/left.csv"
  echo "Video-OpenFace-c   " & wc -l "${BASE_INPUT_DIR}/Video-OpenFace/${study_idx}/center.csv"
  echo "Video-OpenFace-r   " & wc -l "${BASE_INPUT_DIR}/Video-OpenFace/${study_idx}/right.csv"
  echo ""
  echo "Raw Video-OpenFace-l   " & wc -l "${BASE_INPUT_DIR}/Raw_Output_CSVs/Raw_Face2/${study_idx}/left.csv"
  echo "Raw Video-OpenFace-c   " & wc -l "${BASE_INPUT_DIR}/Raw_Output_CSVs/Raw_Face2/${study_idx}/center.csv"
  echo "Raw Video-OpenFace-r   " & wc -l "${BASE_INPUT_DIR}/Raw_Output_CSVs/Raw_Face2/${study_idx}/right.csv"
  
done
