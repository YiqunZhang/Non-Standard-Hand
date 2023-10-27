# Webapp of Detecting and Restoring Non- Standard Hands in Stable Diffusion Generated Images

## Authors

Author: Yiqun Zhang, u7102332

Supervisors: Dr. Zhenyue Qin and Dr. Dylan Campbell

## Getting Started

We have deployed an online version, which you can access at http://gradio.yiqun.io. Alternatively, you can follow the guide below for local deployment.


### Prerequisites

This project is tested on Ubuntu 22.04, Python 3.11 and CUDA 12.2. To ensure sufficient GPU memory, it's recommended to use an RTX 3090.

### Download Model

Download model here: https://1drv.ms/u/s!AucDubvWzmnsjJx5DoZDX9bM8h2yog?e=cDiq3Q

The provided link leads to a zip archive. After downloading, please extract its contents.

Put downloaded and unzipped model into the following directory structure:

```
- webapp/
  - model/
    - ip2p/
    - controlnet-depth-sdxl-1.0/
    - stable-diffusion-xl-base-1.0/
    - yolo.pt
```

### Setting Up the Environment

At this juncture, make sure your shell is currently in the 'webapp' directory.

If you haven't installed `python3-venv` yet, you can do so with the following command:

```bash
sudo apt-get install python3-venv
```

Then, create a virtual environment:

```bash
python -m venv venv
```

And you can activate the virtual environment:

```bash
source venv/bin/activate
```

Then you need to install the dependencies:

```bash
pip install -r requirements.txt
```

### Running the Webapp

To run the webapp, you can use the following command:

```bash
python main.py
```

Then you can visit the webapp at http://localhost:7860/