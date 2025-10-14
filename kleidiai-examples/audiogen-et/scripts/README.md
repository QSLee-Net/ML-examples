<!--
    SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# Building and Running the Audio Generation Application on Arm® CPUs with the Stable Audio Open Small Model

## Goal
This guide will show you how to convert the Stable Audio Open Small Model to ExecuTorch format to run on Arm® CPUs.

### Converting the Stable Audio Open Small Model to ExecuTorch format
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
Clone ExecuTorch repository and run the installation script (tested on `57a79037b1973836705527a59c789de7a0348152`)
```bash
git clone https://github.com/pytorch/executorch.git
git checkout 57a79037b1973836705527a59c789de7a0348152
cd executorch
bash ./install_executorch.sh
```

#### Step 3
Install Stable Audio Open tools dependency
```bash
pip install git+https://github.com/Stability-AI/stable-audio-tools.git@31932349d98c550c48711e7a5a40b24aa3d7c509
```

#### Step 4
Export the model using the following script
```bash
python ./scripts/export_sao.py --ckpt_path model.ckpt --model_config model_config.json
```

> [!NOTE]
>
> If you faced the following issue while converting the model:
> ```bash
> self._files[name] = importlib.resources.read_binary(__package__, name)
>  File "/usr/lib/python3.10/importlib/resources.py", line 88, in read_binary
>    with open_binary(package, resource) as fp:
>  File "/usr/lib/python3.10/importlib/resources.py", line 46, in open_binary
>    return reader.open_resource(resource)
>  File "/usr/lib/python3.10/importlib/abc.py", line 433, in open_resource
>    return self.files().joinpath(resource).open('rb')
>  File "/usr/lib/python3.10/pathlib.py", line 1119, in open
>    return self._accessor.open(self, mode, buffering, encoding, errors,
>FileNotFoundError: [Errno 2] No such file or directory: '../kleidiai-examples/audiogen-et/scripts/executorch/exir/_serialize/program.fbs'
> ```
> You can run the following commands:
> ```bash
> export EXECUTORCH_ROOT=<PATH-TO-EXECUTORCH>
> cp $EXECUTORCH_ROOT/schema/program.fbs $EXECUTORCH_ROOT/exir/_serialize/program.fbs
> cp $EXECUTORCH_ROOT/schema/scalar_type.fbs $EXECUTORCH_ROOT/exir/_serialize/scalar_type.fbs
> ```

The three exported models (`conditioners_model.pte`, `dit_model.pte` and `autoencoder_model.pte`) will be required to run the audiogen application on Android™ device.

You can now follow the instructions located in the [`app/`](../app/README.md) directory to build the audio generation application.