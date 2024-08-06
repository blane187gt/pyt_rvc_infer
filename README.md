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

### **Example Code**

```python
import os
import numpy as np
from scipy.io.wavfile import read, write
from pyt_rvc_infer.lib.infer import infer_audio
from pydub import AudioSegment

inferred_audio = infer_audio(
    model_name=MODEL_NAME,
    audio_path=Audio_Path,
    f0_change=F0_CHANGE,
    f0_method=F0_METHOD,
    min_pitch=MIN_PITCH,
    max_pitch=MAX_PITCH,
    crepe_hop_length=CREPE_HOP_LENGTH,
    index_rate=INDEX_RATE,
    filter_radius=FILTER_RADIUS,
    rms_mix_rate=RMS_MIX_RATE,
    protect=PROTECT,
    split_infer=SPLIT_INFER,
    min_silence=MIN_SILENCE,
    silence_threshold=SILENCE_THRESHOLD,
    seek_step=SEEK_STEP,
    keep_silence=KEEP_SILENCE,
    formant_shift=FORMANT_SHIFT,
    quefrency=QUEFRENCY,
    timbre=TIMBRE,
    f0_autotune=F0_AUTOTUNE,
    output_format=OUTPUT_FORMAT
)

print(f"Inferred audio saved to: {inferred_audio}")

# Load and play the inferred audio
audio_segment = AudioSegment.from_file(inferred_audio)
```
