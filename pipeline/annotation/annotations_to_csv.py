import argparse
import os
import json
import swifter
import pandas as pd


def _get_joined_labels(t, label_key="speaker", annotations=None):
    """Helper for getting the label from the annotations at time t

    Searches through the entire set of annotations to determine all matching
    labels for a given point in time. The matches are joined with the '*' character.

    Args:
        t ([type]): Time in seconds
        label_key (str, optional): Key for pulling the label from the annotations. Defaults to "speaker".
        annotations ([type], optional): List of annotations. Defaults to None.

    Returns:
        [type]: annotation string. If multiple annotations they are concatenated joined with *
    """
    labels = [u[label_key] for u in annotations if t >= u["start"] and t < u["end"]]
    if len(labels) > 1:
        labels = ["*".join(labels)]

    # assert "f" not in labels, f"******Speakers needs fixing!****** time: {t}"

    # Fix label formatting before returning
    if labels:
        return labels[0]
    else:
        return ""


def _get_one_hot(joined_labels, label=None):
    """Helper to iterate through a csv of joined annotations

    Helps to convert a single column of joined_labels to one hot encoding

    Args:
        joined_labels ([type]): [description]
        label ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    try:

        if label in joined_labels:
            return 1
        else:
            return 0
    except Exception as E:
        print(E, label, joined_labels)


def convert_annotations(
    annotation_path,
    output_path,
    frame_rate=30,
    json_key="utterances",
    label_key="speaker",
    feature_csv_path=None,
):
    """Turns annotations that are formatted in .json to CSV

    The CSV is formatted such that each row is a frame and each column is a
    one hot encoded label

    Args:
        annotation_path ([type]): path to annotation json. Json should be formatted as
        a list with individual annotations having a start and stop time and a label.
        output_path ([type]): path to converted csv
        json_key (str, optional): Key to list of annotations. Defaults to "utterances".
        label_key (str, optional): Key to labels that have been annotated. Defaults to "speaker".
        feature_csv_path ([type], optional): Path to feature csv with "timestamps" that will be
        used to create labels that match the features. Defaults to None. If not supplied the timestamps
        will be generated using the framerate.
    """
    with open(annotation_path) as f:
        annotations = json.load(f)[json_key]

    all_labels = [u[label_key] for u in annotations]
    max_time = max([u["end"] for u in annotations])
    unique_labels = list(set(all_labels))
    print("Labels include: \n", unique_labels)
    if feature_csv_path:
        feature_df = pd.read_csv(feature_csv_path, header=0)
        label_df = pd.DataFrame({"timestamp": feature_df["timestamp"]})
    else:
        label_df = pd.DataFrame(
            {"timestamp": [i / 30 for i in range(0, int(max_time * frame_rate + 1))]}
        )

    # Intermediary step creates a single column for all labels
    label_df[label_key] = label_df["timestamp"].swifter.apply(
        _get_joined_labels, label_key=label_key, annotations=annotations
    )

    # Next we convert joined labels into the labels we are interested in
    for label in unique_labels:
        label_df[label] = label_df[label_key].swifter.apply(_get_one_hot, label=label)

    # print(label_df)
    out_df = label_df[unique_labels]
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

    convert_annotations(args.annotation, args.output, feature_csv_path=args.features)
