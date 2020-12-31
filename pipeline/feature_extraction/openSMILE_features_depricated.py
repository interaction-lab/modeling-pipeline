import os
import sys
import time
import argparse


def get_opensmile_features(paths: dict, feature_types: list = ["gemaps"]):
    """High level runner for opensmile

    TODO: increase flexibility of use. Fix egemaps?

    Args:
        paths (dict): contains paths to config, audio_file, csv_dir, and smile_exe
        feature_types (list, optional): [description]. Defaults to ["gemaps"].
    """
    # LOOP THROUGH THE DIFFERENT FEATURE SETS YOU WANT EXTRACTED
    for feature_type in feature_types:
        print(f"extracting for {feature_type}")
        _run_smile(
            _get_configs(feature_type, paths["config"]),
            paths["audio_file"],
            paths["csv_dir"],
            paths["smile_exe"],
        )
    return


def _get_configs(feature_type, path_config):
    """Sorts through configs for different feature sets you can extract with opensmile

    Args:
        feature_type ([type]): Which feature set to use
        path_config ([type]): Path to bas configs

    Returns:
        [type]: everything needed to _run_opensmile()
    """
    if feature_type == "mfcc":
        folder_output = "mfcc_features"  # output folder
        conf_smileconf = "".join(
            [path_config, "MFCC12_0_D_A.conf"]
        )  # MFCCs 0-12 with delta and acceleration coefficients
        opensmile_options = "".join(
            [
                "-configfile ",
                conf_smileconf,
                " -appendcsv 0 -timestampcsv 1 -headercsv 1",
            ]
        )  # options from standard_data_output_lldonly.conf.inc
        output_option = (
            "-csvoutput"  # options from standard_data_output_lldonly.conf.inc
        )

    elif feature_type == "egemaps":
        folder_output = "egemaps_features"  # output folder
        conf_smileconf = "".join(
            [path_config, "gemaps/eGeMAPSv01a.conf"]
        )  # eGeMAPS feature set
        opensmile_options = "".join(
            [
                "-configfile ",
                conf_smileconf,
                " -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1",
            ]
        )  # options from standard_data_output.conf.inc
        output_option = "-lldcsvoutput"  # options from standard_data_output.conf.inc

    elif feature_type == "gemaps":
        folder_output = "gemaps_features"  # output folder
        conf_smileconf = "".join(
            [path_config, "gemaps/GeMAPSv01a.conf"]
        )  # GeMAPS feature set
        opensmile_options = "".join(
            [
                "-configfile ",
                conf_smileconf,
                " -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1",
            ]
        )  # options from standard_data_output.conf.inc
        output_option = "-lldcsvoutput"  # options from standard_data_output.conf.inc
    else:
        print("Error: Feature type " + feature_type + " unknown!")
    return folder_output, conf_smileconf, opensmile_options, output_option


def _run_smile(configs, audio_file, output_dir_path, opensmile_exe_path):
    """Call to opensmile executable with args from _get_configs

    Args:
        configs ([type]): [description]
        audio_file ([type]): [description]
        output_dir_path ([type]): [description]
        opensmile_exe_path ([type]): [description]
    """
    # unload the configs
    output_file, conf_smileconf, opensmile_options, output_option = configs

    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    outfilename = os.path.join(output_dir_path, output_file + ".csv")
    print("will write to: ", outfilename)
    opensmile_call = " ".join(
        [
            opensmile_exe_path,
            opensmile_options,
            "-inputfile ",
            audio_file,
            output_option,
            outfilename,
            "-output ?",
        ]
    )  # (disabling htk output)
    os.system(opensmile_call)
    time.sleep(0.01)
    return


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Produce OpenSmile Features from audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_file", help="where to find input wav file")
    parser.add_argument("csv_dir", help="path for the output CSVs")
    parser.add_argument(
        "openSMILE_dir",
        default="/home/chris/Programs/opensmile-2.3.0",
        help="where to find opensmile executable",
    )
    args = parser.parse_args()
    return args


def main(args):
    args = parse_args(args)

    # MODIFY THESE BASED ON YOUR SYSTEM
    # feature_types = ["egemaps", "gemaps"]
    feature_types = ["gemaps"]

    paths = {
        "audio_file": args.audio_file,
        "csv_dir": args.csv_dir
    }

    # MODIFY this path to the folder of the SMILExtract (version 2.3) executable
    paths["smile_exe"] = os.path.join(args.openSMILE_dir, "SMILExtract")

    # MODIFY this path to the config folder of opensmile 2.3 - no POSIX here on cygwin (windows)
    paths["config"] = os.path.join(args.openSMILE_dir, "config/")

    get_opensmile_features(paths, feature_types)


if __name__ == "__main__":
    main(sys.argv[1:])
