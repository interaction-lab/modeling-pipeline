# Credit for most of this example to:
# https://github.com/wiseman/py-webrtcvad

import collections
import contextlib
import sys
import json
import wave
import os
import argparse
from pipeline.common.file_utils import ensure_destination_exists
import webrtcvad


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    print(path)
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        # TODO: look into videos from an Iphone
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    ensure_destination_exists(path)
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write("1" if is_speech else "0")
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write("Audio found +(%s)" % (ring_buffer[0][0].timestamp,))
                start_time = ring_buffer[0][0].timestamp
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write(" To -(%s)" % (frame.timestamp + frame.duration))
                triggered = False
                yield (
                    b"".join([f.bytes for f in voiced_frames]),
                    start_time,
                    frame.timestamp + frame.duration,
                )
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        pass
        # sys.stdout.write("-(%s)" % (frame.timestamp + frame.duration))
    sys.stdout.write("\n")
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield (
            b"".join([f.bytes for f in voiced_frames]),
            frame.timestamp,
            frame.timestamp + frame.duration,
        )


def get_voice_activity(src_audio: str, aggressiveness: int, dst_dir: str):
    """The voice activity in an audio file is detected and stored.

    Args:
        src_audio (str): path to src audio
        aggressiveness (int): an integer between 0 and 3. 0 is the least aggressive about
                                filtering out non-speech, 3 is the most aggressive.
        dst_dir (str): path to dst for saving json and chunks

    Output:
        The voice activity that is detected is saved as a json. This is useful for
        later annotation, but you may prefer to convert the json to a csv.
    """
    audio, sample_rate = read_wave(src_audio)
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    chunk_folder = os.path.join(dst_dir, "chunks")
    if not os.path.exists(chunk_folder):
        os.makedirs(chunk_folder)
    record = []
    for i, (segment, start, end) in enumerate(segments):
        path = "%s/%002d.wav" % (
            chunk_folder,
            i,
        )
        print(" Writing %s %s %s" % (path, start, end))
        start_string = f"{int(start/60)}:{start%60:.01f}"
        end_string = f"{int(end/60)}:{end%60:.01f}"
        instance = {
            "chunk": path,
            "start": start,
            "end": end,
            "start_string": start_string,
            "end_string": end_string,
            "label": "speaking",
        }
        record.append(instance)
        write_wave(path, segment, sample_rate)
    p = os.path.join(dst_dir, "utterances.json")
    ensure_destination_exists(p)
    with open(p, "w") as outfile:
        json.dump({"utterances": record}, outfile)
    return


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="detect voice activity in an audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_file",
        default="/media/chris/M2/1-Raw_Data/Videos/1/audio/audio.wav",
        help="where to find input wav file",
    )
    parser.add_argument(
        "agressiveness",
        default=0,
        help="How aggressively to consider voice activity",
    )
    parser.add_argument("dst_dir", help="where to place the json and chunks")

    args = parser.parse_args()
    return args


def main(args):
    # args = parse_args(args)
    # get_voice_activity(args.audio_file, args.aggressiveness, args.dst_dir)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
