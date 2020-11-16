#!/bin/bash
# see https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments for detail

# SCRIPT_PATH="${HOME}/Library/OpenFace/build/bin/FaceLandmarkVidMulti"
SCRIPT_PATH="${HOME}/Programs/OpenFace/build/bin/FeatureExtraction"
BASE_INPUT_DIR="/media/$(whoami)/M2/1-Raw_Data/Videos"
BASE_OUTPUT_DIR="/media/$(whoami)/M2/2-Processed_Data/Raw_Output_CSVs/Raw_Face2"

for i in 1 3 {5..27}
do
    mkdir -p "${BASE_OUTPUT_DIR}/${i}" ;
    echo "${i} center"
    ${SCRIPT_PATH} -cam_width=1280 -cam_height=720 -f "${BASE_INPUT_DIR}/${i}/center.mp4" -out_dir "${BASE_OUTPUT_DIR}/${i}/"
    echo "${i} left"
    ${SCRIPT_PATH} -cam_width=1280 -cam_height=720 -f "${BASE_INPUT_DIR}/${i}/left.mp4" -out_dir "${BASE_OUTPUT_DIR}/${i}/"
    echo "${i} right"
    ${SCRIPT_PATH} -cam_width=1280 -cam_height=720 -f "${BASE_INPUT_DIR}/${i}/right.mp4" -out_dir "${BASE_OUTPUT_DIR}/${i}/"
done

echo "shutdown..."
shutdown +10
Video resolution: 1280x720
Sample rate: 44100 Hz
Bits per sample: 32
Frame rate: 30