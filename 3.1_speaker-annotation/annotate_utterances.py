import subprocess
import argparse
import os
import json


class PythonTurnAnnotator:
    def __init__(self, args):
        self.args = args
        with open(args.input) as f:
            # See 2.1_feature-extraction/voice_activity.py for
            # expexted json structure
            self.utterances = json.load(f)["utterances"]
        self.video_path = args.video
        # for keypad input
        self.key_map = {"4": "l", "6": "r", "8": "c", "5": "b"}
        self.person_map = {"l": "left", "c": "center", "r": "right"}

    def _play_video(self, start, stop, speed=1.5):
        """Plays a video through the command line

        Args:
            start (float):
            stop (float):
            speed (float, optional): Rate video is played at. Defaults to 1.5.
        """
        # MPV is a convenient, easily installed video player
        #   with the command line functionality we need
        process = subprocess.Popen(
            [
                "mpv",
                f"--start={start}",
                f"--end={stop}",
                f"--speed={speed}",
                self.video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return

    def _get_utterance_label(self, utterance, split=False):
        """Plays an utterance and expects a user label. Expects clean individual
        labels or an 'f' for fixing in future passes.

        Args:
            utterance (dict): An utterance contains a start and a stop time

        Returns:
            str: label input by the user
        """
        person = None
        speed = 2.5
        while person not in ["l", "r", "c", "b", "e", "f"]:
            print(f"Reviewing {utterance['start_string']} -- {utterance['end_string']}")
            self._play_video(utterance["start"], utterance["end"], speed=speed)

            person = input(
                "((l)eft,(r)ight,(c)enter,(b)ot \n (e)mpty,(f)ix later, _replay): \n --"
            )  # Note- you can feel free to modify the labels or specify a keypad

            if person in self.key_map.keys():
                person = self.key_map[person]
            print(f"you input {person}")
            speed = 1
        return person

    def _fixing_utterance_label(self, utterance):
        """Plays an utterance and expects a user label.

        Args:
            utterance (dict): An utterance contains a start and a stop time

        Returns:
            str: label input by the user
        """
        person = "r"
        while person == "r":
            print(f"Reviewing {utterance['start_string']} -- {utterance['end_string']}")
            self._play_video(utterance["start"], utterance["end"])

            person = input(
                f"(c)ont (r)eplay (p)rev (f)ix [split_time] [label] \n {utterance['start_string']} -- "
            )  # Note- you can feel free to modify the labels

            if person in self.key_map.keys():
                person = self.key_map[person]
            print(f"you input {person}")
        return person

    def _write_utterances(self):
        print("Writing outfile")
        with open(self.args.output, "w") as write_file:
            json.dump({"utterances": self.utterances}, write_file)

    def _load_utterances(self):
        with open(args.input) as f:
            # See voice_activity.py for expexted json structure
            self.utterances = json.load(f)["utterances"]

    def fill_labels(self, i=0):
        """This is intended to be a first pass through unlabeled utterances
        will modify the list of utterances in place and write in progress
        updates.

        Args:
            i (int, optional): Starting point in utterance list. Defaults to 0.

        Returns:
            [utterances]:
        """
        while i < len(self.utterances):
            self._load_utterances()
            write = False
            print(f"{i} out of {len(self.self.utterances)}")

            u = self.utterances[i]

            # We should not repeat work if resuming
            if u["speaker"] != "":
                print(f"*******already filled {u['speaker']}*******")
                person = u["speaker"]

            else:
                person = self._get_utterance_label(u)
                if person == "e":
                    self.utterances.pop(i)
                    i -= 1  # list is shortened
                write = True

            u["speaker"] = person

            # write while in progress so we don't lose our work
            if write:
                self._write_utterances()
            i += 1
        return self.utterances

    def check_empty_spaces(self, i=0, min_gap=0.2):
        """Check between voice activity for missed utterances. Insert utterances into 
        gaps if found.

        Args:
            i (int, optional): Starting point in utterance list. Defaults to 0.

        Returns:
            [utterances]:
        """
        while i < len(self.utterances):
            self._load_utterances()
            write = False
            print(f"{i} out of {len(self.utterances)}")

            if i == len(self.utterances) - 1:
                break
            next_u = self.utterances[i + 1]
            u = self.utterances[i]

            if next_u["start"] - u["end"] > min_gap:
                middle_u = {
                    "chunk": u["chunk"],
                    "start": u["end"],
                    "end": next_u["start"],
                    "start_string": u["end_string"],
                    "end_string": next_u["start_string"],
                    "speaker": "",
                }
                person = self._get_utterance_label(middle_u)
                if person == "e":
                    pass
                else:
                    middle_u["speaker"] = person
                    self.utterances.insert(i + 1, middle_u)
                    write = True
            else:
                print("gap too short")

            if write:
                self._write_utterances()
            i += 1
        return self.utterances

    def fix_speaker_labels(self, i=0):
        """Iterate through the labels and correct those that have been marked
        as 'f' for fix. Will request input from the user and then update the 
        utterance list and file.

        Args:
            i (int, optional): Starting point in utterance list. Defaults to 0.

        Returns:
            [utterances]: list of utterances
        """
        while i < len(self.utterances):
            self._load_utterances()
            print(f"{i} out of {len(self.utterances)}")
            write = False
            u = self.utterances[i]
            try:
                if u["speaker"] == "f":
                    p = self._fixing_utterance_label(u)

                    if not p or p == "c":
                        i += 1
                    elif p in ["p", "f"]:
                        i -= 1
                        u = self.utterances[i]
                        print(f"Segment labeled as {u['person']}")
                        self._play_video(u["start"], u["end"])
                        if input("(f)ix this segment?") == "f":
                            self.utterances[i]["person"] = "f"

                    elif p == "e":
                        self.utterances.pop(i)

                    elif len(p) >= 3:
                        p = p.split(" ")
                        if p[0] == "f":
                            if p[1] == "e":
                                self.utterances.pop(i)
                            else:
                                u["speaker"] = p[1]
                                i += 1

                        else:  # p[0] should be the start time

                            start = u.copy()
                            second = u.copy()

                            split = float(p[0].split(":")[0]) * 60 + float(
                                p[0].split(":")[1]
                            )
                            start_person = p[1]
                            # Ensure the input time split is correct
                            if start["start"] > split or start["end"] < split:
                                print("****BAD TIME INPUT*****")
                                i -= 1
                                continue

                            if start_person != "e":
                                start["end_string"] = p[0]
                                start["end"] = split
                                start["speaker"] = start_person
                                self.utterances[i] = start

                                second["start_string"] = p[0]
                                second["start"] = split
                                self.utterances.insert(i + 1, second)

                            else:
                                self.utterances[i]["start_string"] = p[0]
                                self.utterances[i]["start"] = split
                    self._write_utterances()
                else:
                    i += 1
            except Exception as E:
                print(E)
        return self.utterances

    def check_times(self):
        """Iterate through times and check if they are correct. Checks if
        the string and float times match as well as if the end time comes
        after the start time. Prints out errors so they can be found and 
        fixed by manual entry into the json.

        Returns:
            Bool: Returns True if errors were found.
        """
        errors = False
        for i, u in enumerate(self.utterances):
            start_sec_from_string = float(
                u[f"start_string"].split(":")[0]
            ) * 60 + float(u[f"start_string"].split(":")[1])

            if abs(u["start"] - start_sec_from_string) > 0.05:
                print(f"{i} incorrect start: {u['start']} - ({start_sec_from_string})")
                print(u, "\n")
                errors = True

            end_sec_from_string = float(u[f"end_string"].split(":")[0]) * 60 + float(
                u[f"end_string"].split(":")[1]
            )
            if abs(u["end"] - end_sec_from_string) > 0.05:
                print(f"{i} incorrect end: {u['end']} - ({end_sec_from_string})")
                print(u, "\n")
                errors = True

            if start_sec_from_string > end_sec_from_string:
                print("Incorrect order ", u, "\n")
                errors = True
        return errors

    def clean_times(self, i=0):
        """Iterates through utterances and updates float time with the
        value of the string time.

        Args:
            i (ing, optional): Starting point. Defaults to 0.

        Returns:
            [utterances]: list of utterances
        """
        while i < len(self.utterances):
            self._load_utterances()
            print(f"{i} out of {len(self.utterances)}")
            for c in ["start", "end"]:
                sec_from_string = float(
                    self.utterance[i][f"{c}_string"].split(":")[0]
                ) * 60 + float(self.utterance[i][f"{c}_string"].split(":")[1])

                if abs(self.utterance[i][c] - sec_from_string) > 0.05:
                    print(f"{self.utterance[i][c]} - {sec_from_string}")
                    self.utterance[i][c] = sec_from_string
            i += 1
        self._write_utterances()
        return self.utterances

    def make_turns(self):
        """Merges utterances into turns. Removes 'chunk' from utterance.
        Takes series of continuous utterances by the same person and merges
        them into a single turn.

        Returns:
            [utterances]: list of merged utterances
        """
        self.merged_utterances = []
        for i, u in enumerate(self.utterances):
            u.pop("chunk")
            self.merged_utterances.append(u)
            speaker = u["speaker"]
            end = None
            if len(self.utterances) > i + 1:
                while u["speaker"] == self.utterances[i + 1]["speaker"]:
                    end = self.utterances[i + 1]["end"]
                    end_string = self.utterances[i + 1]["end_string"]
                    self.utterances.pop(i + 1)
                    if len(self.utterances) <= i + 1:
                        break
                if end:
                    self.merged_utterances[-1]["end"] = end
                    self.merged_utterances[-1]["end_string"] = end_string
        self.utterances = self.merged_utterances
        print("Writing outfile")
        with open(self.args.output, "w") as write_file:
            json.dump({"utterances": self.utterances}, write_file)
        return self.merged_utterances

    def review_labels(self):
        for u in self.utterances:
            print(f"Reviewing {u['start_string']} -- {u['end_string']}")
            print(f"Speaker is _{u['speaker']}_")
            self._play_video(u["start"], u["end"], speed=3)

            w = input("_continue, (f)ix, or (b)reak? ")

            if not w:
                continue
            elif w == "b":
                break
            else:
                u["speaker"] = w
                self._write_utterances()

        print("finished reviewing")
        return

    def review_person(self, person):
        print(f"Starting with person: {person}")
        if person == "b":
            self.video_path = self.video_path
        else:
            path_parts = self.video_path.split("/")
            new_path = path_parts[:-1] + [f"{self.person_map[person]}.mp4"]
            self.video_path = "/".join(new_path)

        for u in self.utterances:
            if person in u["speaker"]:
                print(f"Reviewing {u['start_string']} -- {u['end_string']}")
                self._play_video(u["start"], u["end"], speed=3)

                w = input("_continue, (f)ix, or (b)reak? ")

                if not w:
                    continue
                elif w == "b":
                    break
                else:
                    u["speaker"] = w
                    self._write_utterances()
        print("finished reviewing")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take unlabeled utterances.json and add labels.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="path to video to play")
    parser.add_argument("input", help="path to input json")
    parser.add_argument("output", help="path to output json")
    parser.add_argument("steps", help="steps to be completed [01234]")
    args = parser.parse_args()
    PA = PythonTurnAnnotator(args)

    if "0" in args.steps:
        PA.fill_labels()

    if "1" in args.steps:
        PA.check_empty_spaces()
        if PA.check_times():
            if input("Clean times? (y/n)") == "y":
                PA.clean_times()

    if "2" in args.steps:
        PA.fix_speaker_labels()
        if PA.check_times():
            if input("Clean times? (y/n)") == "y":
                PA.clean_times()

    if "3" in args.steps:
        while PA.check_times():
            input(
                "errors found, continue? (You can fix errors now and they will be reloaded)"
            )
            with open(args.input) as f:
                PA.utterances = json.load(f)["utterances"]
        print("Times correct, merging.")
        PA.make_turns()
        if PA.check_times():
            if input("Clean times? (y/n)") == "y":
                PA.clean_times()

    if "4" in args.steps:
        for p in ["l", "r", "c", "b"]:
            PA.review_person(p)
            PA.video_path = args.video  # reset back
        PA.review_labels()
        if PA.check_times():
            if input("Clean times? (y/n)") == "y":
                PA.clean_times()
