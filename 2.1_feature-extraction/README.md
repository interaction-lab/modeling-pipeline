# Feature Extraction

## Supported Libraries

- Librosa
- OpenFace
- OponSMILE
- py-webrtcvad (voice activity)

## Expected Format

Video should be recorded in the MP4 format and audio should be seperate wav files. You can convert files to these formats using example utility scripts in the 0.0_file-manipulation folder. 

Features can be extracted at any frame or sample rate (FPS and SR respectively). In order for the audio and video features to match rates, the video should be extracted at 30 fps and the audio will match (using a hop length equal to 30 divided by the Sample Rate). In the future this will be parameterized to match audio features to any video frame rate.

## ToDo

See the main README.md for areas to work on to improve the feature extraction step of the pipeline.