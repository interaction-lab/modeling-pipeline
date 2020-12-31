import os
import sys
import time
import argparse
import opensmile
from pipeline.common.file_utils import ensure_destination_exists
import audiofile


def get_opensmile_features(src_audio: str, dst_csv: str):
    """Extract audio features from audio file

    TODO: These should be set to produce audio features at the 30fps matching most video

    Args:
        src_audio (str): [description]
        dst_csv (str): [description]

    Returns:
        [type]: [description]
    """
    signal, sampling_rate = audiofile.read(src_audio, always_2d=True)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv01b,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
    )
    opensmile.FeatureLevel.LowLevelDescriptors
    # y = smile.process_file(src_audio)
    y = smile.process_signal(signal, sampling_rate)

    ensure_destination_exists(dst_csv)
    y.to_csv(dst_csv, index=False)
    return y


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="produce opensmile egemaps audio features for an audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_file",
        default="/media/chris/M2/1-Raw_Data/Videos/1/audio/audio.wav",
        help="where to find input wav file",
    )
    parser.add_argument("csv_path", help="where to place the CSV")
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args(args)
    df = get_opensmile_features(args.audio_file, args.csv_path)


if __name__ == "__main__":
    main(sys.argv[1:])
