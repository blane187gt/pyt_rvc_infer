# PYT-RVC-INFER
--- 

## **Credits**

IAHispano
RVC-Project
FFmpeg
yt_dlp
etc

## **How to Use**

### **Inference**

Configure the following basic and advanced settings in your script:

#### **Basic Settings**

- `MODEL_NAME` (str): Name of the model. The system will search for the corresponding folder containing the `.pth` and `.index` files.
- `Audio_Path` (str): Path to the audio file you want to process.
- `F0_CHANGE` (int): Change in pitch (in semitones).
- `F0_METHOD` (str): Method for F0 (pitch) estimation. Options: `"hybrid[rmvpe+fcpe]"`.
- `MIN_PITCH` (str): Minimum pitch value.
- `MAX_PITCH` (str): Maximum pitch value.
- `CREPE_HOP_LENGTH` (int): Hop length for CREPE (a pitch estimation tool).
- `INDEX_RATE` (float): Rate for index calculations.
- `FILTER_RADIUS` (int): Radius for filtering.
- `RMS_MIX_RATE` (float): Rate for RMS mixing.
- `PROTECT` (float): Protect value.

#### **Advanced Settings**

- `SPLIT_INFER` (bool): Split input audio into smaller chunks based on silence detection.
- `MIN_SILENCE` (int): Minimum length of silence (in milliseconds) to detect.
- `SILENCE_THRESHOLD` (int): Upper bound for detecting silence (in dBFS).
- `SEEK_STEP` (int): Step size for iterating over audio (in milliseconds).
- `KEEP_SILENCE` (int): Amount of silence to retain at the start and end of chunks (in milliseconds).
- `FORMANT_SHIFT` (bool): Experimental setting for better male-to-female voice conversion.
- `QUEFRENCY` (float): Controls the rate of frequency change.
- `TIMBRE` (float): Controls the "sharpness" of the audio.
- `F0_AUTOTUNE` (bool): Autotune to the closest note frequency.

#### **Output Settings**

- `OUTPUT_FORMAT` (str): Desired format for the output audio file (e.g., "wav").



## tutorial code will added comming so soon!
