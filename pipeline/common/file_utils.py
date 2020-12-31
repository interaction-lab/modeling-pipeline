import os
import shutil
import pathlib
import fnmatch
import yaml


def ensure_destination_exists(dst_file_path: str):
    """Creates a folder for the file

    Will create the destination folder from the given file path without the
    the file name. Use with care as this could have unintended consequences
    if you accidently use a partial file path.
    TODO: add some checks that make sense to the path being created

    Args:
        dst_file_path (str): Full path of a file
    """
    dst_dir = "/".join(dst_file_path.split('/')[:-1])
    if not os.path.isdir(dst_dir):
        pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)
    return


def move_dir(src: str, dst: str, pattern: str = '*'):
    """Move files from src to dst if they match patterns

    Args:
        src (str path): directory
        dst (str path): directory
        pattern (str, optional): pattern describing files to match (e.g. *.csv). Defaults to '*'.
    """

    # this is a useful line for making directories if they don't exist
    if not os.path.isdir(dst):
        pathlib.Path(dst).mkdir(parents=True, exist_ok=True)

    for f in fnmatch.filter(os.listdir(src), pattern):
        shutil.move(os.path.join(src, f), os.path.join(dst, f))
    return


def get_dirs_from_config(config: str = "./config/dir_sample_config.yml"):
    """Creates a list of directory paths from a config.yml

    Args:
        config (str, optional): config path. Must contain 'dir_pattern' and
        'substitutes' keys. Defaults to "./config/dir_sample_config.yml".

    Returns:
        [paths]: paths generated from the config
    """
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dir_list = [os.path.join(*config["dir_pattern"])]

    for sub_dir, subs in config["substitutes"].items():
        assert sub_dir in config["dir_pattern"], f"substitution ({sub_dir}) must have a target match in the path {config['dir_pattern']}"
        new_dir_list = []
        for p in dir_list:
            for new_dir in subs:
                new_dir_list.append(p.replace(sub_dir, str(new_dir), 1))
        dir_list = new_dir_list
    return dir_list


def move_files(src_config: str = "./config/dir_src_config.yml", dst_config: str = "./config/dir_dst_config.yml", pattern="*"):
    """Moves files from src to dst as described in config patterns

    Uses get_dirs_from_config and move_dir to move files on any OS.

    Args:
        src_config (str, optional): Defaults to "./config/dir_src_config.yml".
        dst_config (str, optional): Defaults to "./config/dir_dst_config.yml".
        pattern (str, optional): pattern used to identify files to move. Defaults to "*".
    """
    s = get_dirs_from_config(src_config)
    d = get_dirs_from_config(dst_config)
    assert len(s) == len(d), "must have the same number of src and dst"
    for i in range(len(s)):
        move_dir(s[i], d[i], pattern)
    return


def get_file_length(file_path: str = "./file_utils.py"):
    """Efficiently get length of file

    Args:
        file_path (str, optional): Path of file to read. Defaults to "./file_utils.py".

    Returns:
        int: length of file
    """

    sys_call = f"wc -l {file_path}"
    wc_return = os.popen(sys_call).read()
    file_len = int(wc_return.split()[0]) + 1
    return file_len


def zip_files(files_to_zip: list, dst_path: str):
    """Zip files together.

    Useful for compressing for upload or download

    Args:
        files_to_zip (list): list containing either a single file/dir or
        multiple files or directories
        dst_path (str): path of output zip file
    """
    ensure_destination_exists(dst_path)
    sys_call = f"zip -r {dst_path} {' '.join(files_to_zip)}"
    os.popen(sys_call)
    return


def resample_audio(src_audio: str, dst_audio: str, rate: int):
    """Convert audio rate.

    Args:
        src_audio (str): src audio file path
        dst_audio (str): dst audio file path
        rate (int): new sample rate for dst audio
    """
    ensure_destination_exists(dst_audio)
    sys_call = f"ffmpeg -i {src_audio} -ar {rate} {dst_audio}"
    os.popen(sys_call)
    return


def audio_from_video(src_video: str, dst_audio: str):
    """Pull audio from source mp4 to destination wav.

    to compress the audio into mp3 add the "-map 0:a" before the
    destination file name. The

    Args:
        src_video (str): Path to src video
        dst_audio (str): Path to dst audio
    """
    ensure_destination_exists(dst_audio)
    sys_call = f"ffmpeg -i {src_video} -ac 1 {dst_audio}"
    os.popen(sys_call)
    return


def main():
    """If you would like to use any of the utility functions provided here
        independently, just uncomment the appropriate function, set the configuration
        for you computer, and run this file.
    """
    # get_dirs_from_config("./config/dir_src_config.yml")
    # move_files(src_config="./config/dir_src_config.yml", dst_config="./config/dir_dst_config.yml", pattern="*.json")
    # get_file_length()
    # files_to_zip = ["./bash_utils/count_csv_lines.sh", "./bash_utils/move_to_folder.sh", "./bash_utils/reorganize_files.sh"]
    # files_to_zip = ["./bash_utils"]
    # zip_files(files_to_zip, "./test.zip")
    pass


if __name__ == "__main__":
    main()
