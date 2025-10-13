<!--
    SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Building and Running the Audio Generation Application on Arm® CPUs with the Stable Audio Open Small Model

## Dependencies
- A host laptop/PC with a Linux®-based operating system (tested on Ubuntu® 20.04.4 LTS with x86_64) or with macOS®.
- The Android™ NDK r27c, which can be downloaded from [here](https://developer.android.com/ndk/downloads).
- CMake version: 3.16.0 or above, which can be downloaded from [here](https://cmake.org/download/)
- The three modules of Stable Audio Open Small model (T5, DiT, and AutoEncoder) converted into ExecuTorch format.
- You can follow the instructions provided in the `scripts/` folder’s [README.md](../scripts/README.md) file to generate the models.

## Goal

This guide will show you how to build the <strong>audio generation (audiogen)</strong> app contained in the single <strong>main.cpp</strong> file. Instructions in this guide are provided for running the audiogen app either on an Android™ device or on a reasonably modern platform with macOS®.

## Building the Audio Generation App

To build the audiogen application, follow one the following sections depending on your <strong>TARGET</strong> platform:

- [Build the audiogen app for macOS® (TARGET)](#build-the-audiogen-app-on-macos_host_for-macos_target)
- [Build the audiogen app for Android™ (TARGET)](#build-the-audiogen-app-on-linux_host_or-macos_host_for-android_target)

### Build the audiogen app on macOS® (HOST) for macOS® (TARGET)
#### Step 1
When building the application, make sure you are inside a fresh virtual environment (it is recommended to use Python 3.10 for compatibility with the specified packages):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

#### Step 2
Navigate to the `audiogen-et/app/` folder. Set the `EXECUTORCH_MODELS_PATH` environment variable to the path where your Stable Audio Open Small models exported to ExecuTorch are located:

```bash
export EXECUTORCH_MODELS_PATH=<path_to_your_executorch_models>
```

#### Step 3
Build the audiogen application. Inside the `app` directory, create the `build` folder and navigate into it:

```bash
mkdir build && cd build
```

Next, run CMake using the following command:

```bash
cmake ..
```

Then, build the application:
```bash
make -j
```

#### Step 4
Since the tokenizer used in the audiogen application is based on <strong>SentencePiece</strong>, you’ll need to download the `spiece.model` file from: https://huggingface.co/google-t5/t5-base/tree/main
and add it to your `$EXECUTORCH_MODELS_PATH`.

```bash
curl https://huggingface.co/google-t5/t5-base/resolve/main/spiece.model -o $EXECUTORCH_MODELS_PATH/spiece.model
```

At this point, you are ready to run the audiogen application.

From there, you can then run the `audiogen` application, which requires just three input arguments:

- **Model Path (-m)**: The directory containing your ExecuTorch models and `spiece.model` files
- **Prompt (-p)**: A text description of the desired audio (e.g., *warm arpeggios on house beats 120BPM with drums effect*)
- **CPU Threads (-t)**: The number of CPU threads to use (e.g., `4`)

```bash
./audiogen -m $EXECUTORCH_MODELS_PATH -p "warm arpeggios on house beats 120BPM with drums effect" -t 4
```

If everything runs successfully, the generated audio will be saved in `.wav` format (`warm_arpeggios_on_house_beats_120bpm_with_drums_effect_99.wav`) in the `audiogen-et/app/build` folder. At this point, you can play it on your laptop or PC.

### Build the audiogen app on Linux® (HOST) or macOS® (HOST) for Android™ (TARGET)

#### Step 1
When building the application, make sure you are inside a fresh virtual environment (it is recommended to use Python 3.10 for compatibility with the specified packages):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

#### Step 2
Navigate to the `audiogen-et/app/` folder. Set the `EXECUTORCH_MODELS_PATH` environment variable to the path where your Stable Audio Open Small models exported to ExecuTorch are located:

```bash
export EXECUTORCH_MODELS_PATH=<path_to_your_executorch_models>
```

#### Step 3
If you haven't installed the Android™ NDK r27c yet, download and extract the Android™ NDK r27c in the `app` directory:

```bash
# On Linux®
wget https://dl.google.com/android/repository/android-ndk-r27c-linux.zip
unzip android-ndk-r27c-linux

# On macOS®
curl https://dl.google.com/android/repository/android-ndk-r27c-darwin.zip -o android-ndk-r27c-darwin.zip
unzip android-ndk-r27c-darwin
```

Set the `NDK_PATH` environment variable to the path where you extracted the Android™ NDK r27c:

```bash
export NDK_PATH=$(pwd)/android-ndk-r27c
```
> If you extracted the Android™ NDK to a different directory, be sure to update `NDK_PATH` accordingly.

#### Step 4

Build the audiogen application. Inside the `app` directory, create the `build` folder and navigate into it:

```bash
mkdir build && cd build
```

Next, run CMake using the following command:

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a ..
```

Then, build the application:
```bash
make -j
```
#### Step 5
At this point, you are ready to push the binaries to your Android™ device and run the audiogen application. To do so, use the `adb` tool to push all necessary files into `/data/local/tmp/app`

```bash
adb shell mkdir -p /data/local/tmp/app
adb push audiogen /data/local/tmp/app
adb push $EXECUTORCH_MODELS_PATH/dit_model.pte /data/local/tmp/app
adb push $EXECUTORCH_MODELS_PATH/autoencoder_model.pte /data/local/tmp/app
adb push $EXECUTORCH_MODELS_PATH/conditioners_model.pte /data/local/tmp/app
```

Since the tokenizer used in the audiogen application is based on <strong>SentencePiece</strong>, you’ll need to download the `spiece.model` file from:
https://huggingface.co/google-t5/t5-base/tree/main
and transfer it to your device.

```bash
# On Linux®
wget https://huggingface.co/google-t5/t5-base/resolve/main/spiece.model
adb push spiece.model /data/local/tmp/app

# On macOS®
curl https://huggingface.co/google-t5/t5-base/resolve/main/spiece.model -o spiece.model.zip
adb push spiece.model /data/local/tmp/app
```

At this point, you are ready to test the audiogen application.

Use the `adb` tool to enter the device:

```bash
adb shell
```

Then, go to `/data/local/tmp/app`
```bash
cd /data/local/tmp/app
```

From there, you can then run the `audiogen` application, which requires just three input arguments:

- **Model Path (-m)**: The directory containing your ExecuTorch models and `spiece.model` files
- **Prompt (-p)**: A text description of the desired audio (e.g., *warm arpeggios on house beats 120BPM with drums effect*)
- **CPU Threads (-t)**: The number of CPU threads to use (e.g., `4`)

```bash
./audiogen -m . -p "warm arpeggios on house beats 120BPM with drums effect" -t 4
```

If everything runs successfully, the generated audio will be saved in `.wav` format (`warm_arpeggios_on_house_beats_120bpm_with_drums_effect_99.wav`) in the same directory as the `audiogen` binary. At this point, you can then retrieve it using the `adb` tool from a different Terminal and play it on your laptop or PC.

```bash
adb pull data/local/tmp/warm_arpeggios_on_house_beats_120bpm_with_drums_effect_99.wav
```
