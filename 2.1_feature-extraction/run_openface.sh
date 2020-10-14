#!/bin/bash
# see https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments for detail

# SCRIPT_PATH="${HOME}/Library/OpenFace/build/bin/FaceLandmarkVidMulti"
SCRIPT_PATH="${HOME}/Library/OpenFace/build/bin/FeatureExtraction"
# BASE_INPUT_DIR="${HOME}/Desktop/Chris Study"
BASE_INPUT_DIR="/media/$(whoami)/MP-HRI Backup/Study/Data"
BASE_OUTPUT_DIR="${HOME}/Chris Study/Features/Face Feature"

for i in {16..27}
do
    mkdir -p "${BASE_OUTPUT_DIR}/${i}" ;
    # echo "${SCRIPT_PATH} -f \"${BASE_INPUT_DIR}/${i}/center.mp4\" -f \"${BASE_INPUT_DIR}/${i}/left.mp4\" -f \"${BASE_INPUT_DIR}/${i}/right.mp4\" -out_dir \"${BASE_OUTPUT_DIR}/${i}/\""
    ${SCRIPT_PATH} -f "${BASE_INPUT_DIR}/${i}/center.mp4" -f "${BASE_INPUT_DIR}/${i}/left.mp4" -f "${BASE_INPUT_DIR}/${i}/right.mp4" -out_dir "${BASE_OUTPUT_DIR}/${i}/"
done

echo "shutdown..."
shutdown +10
