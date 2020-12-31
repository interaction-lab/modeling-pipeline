import sys
import os
import argparse


def get_openface_features(
    openface_exe: str,
    src_video: str,
    out_dir: str,
    cam_width: int = 1280,
    cam_height: int = 720,
):
    """Extract facial features with OpenFace



    Args:
        openface_exe (str): path to OpenFace executable (OpenFace/build/bin/FeatureExtraction)
        src_video (str): path to video file to get features from
        out_dir (str): path to destination to put features and other outputs
        cam_width (int, optional): [description]. Defaults to 1280.
        cam_height (int, optional): [description]. Defaults to 720.
    """
    # see https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments for detail
    sys_call = f"{openface_exe} -cam_width={cam_width} -cam_height={cam_height} -f {src_video} -out_dir {out_dir}"
    print(sys_call)
    os.popen(sys_call)
    return


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="produce openface video features for an video file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "openface_exe",
        default="/home/chris/Programs/OpenFace/build/bin/FeatureExtraction",
        help="where to find input mp4 file",
    )
    parser.add_argument(
        "video_file",
        default="/media/chris/M2/1-Raw_Data/Videos/1/left.mp4",
        help="where to find input video file",
    )
    parser.add_argument(
        "out_dir",
        default="/media/chris/M2/1-Raw_Data/RawOpenFace/1/",
        help="where to place the feature csv",
    )

    args = parser.parse_args()
    return args


def main(args):
    args = parse_args(args)
    df = get_openface_features(args.video_file)


if __name__ == "__main__":
    main(sys.argv[1:])
