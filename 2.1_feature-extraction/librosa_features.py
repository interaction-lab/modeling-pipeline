import librosa
import sys
import argparse
import numpy as np
import pandas as pd


def get_features(audio_file):
    y, sr = librosa.load(audio_file)
    hop_length = int(sr / 30)  # gives 1 feature per frame
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


def write_csv(features, path):
    df = pd.concat(
        [pd.DataFrame(v) for k, v in features.items()],
        axis=0,
        keys=list(features.keys()),
    ).T
    df.columns = [f"{f}_{s}" if s != "" else f"{f}" for f, s in df.columns]
    df.to_csv(path, index=False)


def main(args):
    parser = argparse.ArgumentParser(
        description="take clean labels.json and produce turns.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_file",
        default="/media/chris/M2/1-Raw_Data/Videos/1/audio/audio.wav",
        help="where to find input wav file",
    )
    parser.add_argument("csv_path", help="where to place the CSV")
    args = parser.parse_args()

    features = get_features(args.audio_file)
    write_csv(features, args.csv_path)


if __name__ == "__main__":
    main(sys.argv[1:])
