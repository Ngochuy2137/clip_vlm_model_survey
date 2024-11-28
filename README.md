# clip_vlm_model_survey
This repo is for surveying performance of a VLM model - CLIP

## Installation
- Follow installation instruction: https://github.com/openai/CLIP

### Recommendation
- Install conda environment with python 3.10
    
    ```bash
    conda create -n clip_env python=3.10
    ```
    
- Check your cuda version
    ```bash
    nvidia-smi
    ```
    - If your cuda version is 12, you can use cudatoolkit 11 or later version
- Install other dependencies
    ```bash
    conda install --yes -c pytorch pytorch torchvision cudatoolkit=11.0
    ```