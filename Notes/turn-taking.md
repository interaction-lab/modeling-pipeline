## Data Story (Chain of Custody)

[X] Review Synch of Audio and Video and Crop Videos to Appropriate Length
- Take a look at videos with waveforms in OpenShot Video Editor
  - Crop start and end of videos
  - Synch waveform with video
  - Save seperate and joint video/audio

[X] We start with the raw original data:
- Videos
- ROS Bags (Depth Recordings)
- User Annotations
- Surveys

[X] We then generate the processed data, using tools such as openface, openpose, google stt and librosa
- raw utterances (voice_activity.py)
- raw face
- raw pose
- raw transcripts
- raw audio features

[X] We then process these data into useful features
- speaker (label_speakers.py, fix_speakers.py, then merge_speakers.py)
- face
- ~~pose~~
- clean transcripts (See Annotation Templates)
- audio features (Pitch and Power)

[X] We then annotate for additional information
- Annotation
    - Speaker
    - Adressee
    - Share Type & Quality

[  ] We inspect and analyze our data and features
- See multi-party-analysis

[  ] We generate features files for use with our models
- generate HDF5 files
- See multi-party-modeling for where this story continues