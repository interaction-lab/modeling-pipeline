#!/bin/bash

# This script will zip together the openface results for the 3 open face runs for each session

BASE_INPUT_DIR="${HOME}/Chris Study/Features/Face Feature"
BASE_OUTPUT_DIR="/media/$(whoami)/MP-HRI Backup/Study Feature/Open Face"

for i in {1..3} {5..27}
do
    # echo $i
    input_dir="${BASE_INPUT_DIR}/${i}"
    output_dir="${BASE_OUTPUT_DIR}/${i}"

    mkdir "${output_dir}" ;
    for direction in "center" "group" "left" "right"
    do
        cd "${input_dir}" && \
        zip -r "${output_dir}/${direction}_aligned.zip" "${direction}_aligned/"
    done
done

echo "shutdown..."
shutdown +10
