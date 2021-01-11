@REM set BASE_INPUT_DIR="C:\Users\klein\Desktop\infant_robot\wav\Side"
@REM set BASE_OUTPUT_DIR="C:\Users\klein\Desktop\infant_robot\wav\Side"
@REM mkdir %BASE_OUTPUT_DIR%\1
@REM python librosa_features.py C:/Users/klein/Desktop/infant_robot/wav/Side/td1.wav C:\Users\klein\Desktop\infant_robot\audio_features\librosa\1\features.csv

@echo off 
set list=1 2 3 4 5 6 7 8 9 10 11 12 13 
(for %%a in (%list%) do (
   echo %%a
   mkdir C:\Users\klein\Desktop\infant_robot\audio_features\NSF1_librosa\TD%%a
   python librosa_features.py C:/Users/klein/Desktop/infant_robot/wav/NSF1/td%%a.wav C:\Users\klein\Desktop\infant_robot\audio_features\NSF1_librosa\TD%%a\features.csv

   echo/

))


set list=5 7 9 11 13
(for %%a in (%list%) do (
   echo %%a
   mkdir C:\Users\klein\Desktop\infant_robot\audio_features\NSF2_librosa\TD%%a
   python librosa_features.py C:/Users/klein/Desktop/infant_robot/wav/NSF2/td%%a.wav C:\Users\klein\Desktop\infant_robot\audio_features\NSF2_librosa\TD%%a\features.csv

   echo/

))