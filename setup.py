import os

import platform

import pkg_resources

from setuptools import find_packages, setup



setup(

    name="pyt_rvc_infer",

    version="1.0.1",

    description="Python wrapper for simple  inference with rvc v2",

    long_description=open('README.md').read(),

    long_description_content_type='text/markdown',

    readme="README.md",

    python_requires=">=3.10",

    author="blane187gt",

    url="https://github.com/blane187gt/pyt_rvc_infer",

    license="MIT",

    packages=find_packages(),

    package_data={'': ['*.txt', '*.rep', '*.pickle']},

    install_requires=[
      "deemix",
      "fairseq",
      "faiss-cpu",
      "ffmpeg-python",
      "gradio",
      "librosa",
      "numpy"
      "audio-separator[gpu]",
      "scipy",
      "yt_dlp",
      "onnxruntime-gpu",
      "praat-parselmouth",
      "pedalboard",
      "pydub",
      "pyworld",
      "requests",
      "soundfile",
      "torch",
      "torchcrepe",
      "tqdm",
      "torchfcpe",
      "local_attention",
      "ffmpeg",
      "pyworld",
      "torchfcpe",
      "sox",
      "av",
      "pydub",
    ],

    include_package_data=True,
)