<!--
    SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->


# Introduction

The Stable Audio Open Small Model is made of three submodules: Conditioners (Text conditioner and number conditioners), a diffusion transformer (DiT), and an AutoEncoder:
* Conditioners: Consist of T5-based text encoder for the input prompt and a number conditioner for total seconds input. The conditioners encode the inputs into numerical values to be passed to DiT model.
* Diffusion transformer (DiT): It takes a random noise, and denoises it through a defined number of steps, to resemble what the conditioners intent.
* AutoEncoder: It compresses the input waveforms into a manageable sequence length to be processed by the DiT model. At the end of de-noising step, it decompresses the result into a waveform.

# Instructions

## Step 1: Setup
> :warning: **double check your python environment**: make sure `conda activate <VENV>` is run before all the bash and python scripts.

Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch. For installation run `./install_executorch.sh`

## Step 2: Prepare model
1. Install the model dependency:
```
pip install stable-audio-tools==0.0.19
```
2. Export the model:
```
python ./export_sao.py --ckpt_path model.ckpt --model_config model_config.json
```

# Benchmarking
You can benchmark each model using `executor_runner` utility:
```
/executor_runner --model_path <model_path> 
```