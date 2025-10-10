<!--
    SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Building and Running the Audio Generation Application on Arm® CPUs with the Stable Audio Open Small Model

## Goal
This guide will show you how to convert the Stable Audio Open Small Model to ExecuTorch form to run on Arm® CPUs.

### Converting the Stable Audio Open Small Model to LiteRT format
The Stable Audio Open Small Model is made of three submodules:
- Conditioners (Text conditioner and number conditioners)
- Diffusion Transformer (DiT)
- AutoEncoder.

### Create a virtual environment and install dependencies.

#### Step 1
In the `/audiogen-et` folder, create and activate a virtual environment (it is recommended to use Python 3.10 for compatibility with the specified packages)
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

#### Step 2
Clone ExecuTorch repository and run the installation script
```bash
git clone https://github.com/pytorch/executorch.git
bash ./install_executorch.sh
```

### Step 3
Install Stable Audio Open tools dependency
```bash
pip install stable-audio-tools==0.0.19
```

### Step 4
Export the model using the following script
```bash
python ./scripts/export_sao.py --ckpt_path model.ckpt --model_config model_config.json
```

The three exported models will be required to run the audiogen application on Android™ device.

You can now follow the instructions located in the [`app/`](../app/README.md) directory to build the audio generation application.