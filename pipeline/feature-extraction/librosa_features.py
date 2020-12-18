import librosa
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib as plt


def get_librosa_features(src_audio: str):
    """Extract basic audio features from an audio file for HRI with Librosa

    TODO: Allow specification of which librosa features to extract.

    Args:
        src_audio (str): Path to src audio

    Returns:
        df (DataFrame): Audio features dataframe with features as columns
        with the format [feature type]_[number]
    """

    y, sr = librosa.load(src_audio)
    hop_length = int(sr / 30)  # gives 1 feature per frame
    features = _get_features(y, sr, hop_length)
    # _plot_features(features["MFCC"]) # Example plot
    df = _features_to_df(features)
    return df


def _get_features(y: np.ndarray, sr: int, hop_length: int):
    """Extracts audio features with the librosa library.

    Currently set up to get MFCC, Chroma, Mel_Spect, and Rolloff as 
    these features have been identified as well suited for use with 
    machine learning on human audio data. Tonnetz is excluded because
    it doesn't produce the same vector length as the others.

    Args:
        y (np.ndarray): Input audio wave form
        sr (int): Sample Rate
        hop_length (int): Hop Length dictates the number of features per frame
        and is calculated by dividing the sample rate by the desired number of
        features per frame. (e.g. sr/30 gives 30 features per frame)

    Returns:
        dictionary: a dictionary of feature types
    """
    features = {
        "MFCC": librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length),
        "Chroma": librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length),
        "Mel_Spect": librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length),
            ref=np.max,
        ),
        "Spect_Contrast": librosa.feature.spectral_contrast(
            S=np.abs(librosa.stft(y, hop_length=hop_length)),
            sr=sr,
            hop_length=hop_length,
        ),
        # "Tonnetz":librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr, hop_length=704),
        "Rolloff": librosa.power_to_db(
            librosa.feature.spectral_rolloff(
                y=y, sr=sr, hop_length=hop_length, roll_percent=0.95
            ),
            ref=np.max,
        ),
    }
    return features


def _features_to_df(features):
    """Converts dictionary of audio features to df

    Args:
        features (dictionary): dictionary of features

    Returns:
        df (pd.DataFrame): standard dataframe of features
    """
    df = pd.concat(
        [pd.DataFrame(v) for k, v in features.items()],
        axis=0,
        keys=list(features.keys()),
    ).T
    df.columns = [f"{f}_{s}" if s != "" else f"{f}" for f, s in df.columns]
    return df


def _plot_features(features, name):
    plt.figure(figsize=(20, 4))
    librosa.display.specshow(features, x_axis='time')
    plt.colorbar()
    plt.title(name)
    plt.tight_layout()
    print(np.amax(features), np.amin(features), features.shape)
    plt.show()


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="produce librosa audio features for an audio file",
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
    df = get_librosa_features(args.audio_file)
    df.to_csv(args.csv_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
