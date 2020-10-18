import argparse
import os
import json
import swifter
import pandas as pd


def get_speakers(t, utterances=None):
    speakers = [u["speaker"] for u in utterances if t >= u["start"] and t < u["end"]]
    if len(speakers) > 1:
        speakers = ["".join(speakers)]
    assert "f" not in speakers, f"Speakers needs fixing!!!!!!!!!!! {t}"
    if speakers:
        return speakers[0]
    else:
        return ""


def get_speaker(speakers, speaker=None):
    try:
        if speaker[0] in speakers:
            return 1
        else:
            return 0
    except Exception as E:
        print(E, speaker, speakers)


def convert_annotations(annotation_path, output_path, feature_csv=None):
    with open(annotation_path) as f:
        utterances = json.load(f)["utterances"]
    if feature_csv:
        df = pd.read_csv(feature_csv, header=0)
        time_df = pd.DataFrame({"timestamp": df["timestamp"]})

        time_df["speakers"] = time_df["timestamp"].swifter.apply(
            get_speakers, utterances=utterances
        )

        for p in ["left", "center", "right", "bot"]:
            time_df[p] = time_df["speakers"].swifter.apply(get_speaker, speaker=p)
    # print(time_df)
    out_df = time_df[["left", "center", "right", "bot"]]
    out_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take turns.json and creates a csv of labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("annotation", help="path to video to play")
    parser.add_argument("output", help="path to output json")
    parser.add_argument("features", help="path to matching (cleaned) features csv")
    args = parser.parse_args()

    convert_annotations(args.annotation, args.output, feature_csv=args.features)
