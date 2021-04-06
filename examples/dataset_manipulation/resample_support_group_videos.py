

from pipeline.common.file_utils import convert_video, get_dirs_from_config


s = get_dirs_from_config("/home/chris/code/modeling-pipeline/examples/dataset_manipulation/config/resample/src.yml")
d = get_dirs_from_config("/home/chris/code/modeling-pipeline/examples/dataset_manipulation/config/resample/dst.yml")
assert len(s) == len(d), "must have the same number of src and dst"

for i in range(len(s)):
    convert_video(s[i], d[i], 16000, 25)
    