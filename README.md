# OpenCLIP: Zero-Shot and Linear Probe Evaluation of CLIP ViT-B-32 on CIFAR-10

This project is based on the CLIP (Contrastive Language-Image Pre-training) model introduced by Radford et al. [2021](https://arxiv.org/abs/2103.00020). We evaluate the **CLIP ViT-B-32** model for zero-shot image classification and linear probe classification on the **CIFAR-10** dataset.

## Objective
The main objectives of this project are:
1. **Zero-shot evaluation** using the CLIP ViT-B-32 model on CIFAR-10 with various text descriptions.
2. **Linear probe evaluation** after training the model using logistic regression with hyperparameter tuning.
3. **Exploring text template variations** to analyze how different textual descriptions affect zero-shot performance.

## Project Files

Here is the directory structure of the project:

```
OpenCLIP_Image_Classification/
│
├── cifar10_evaluation/           # Folder containing evaluation scripts
│   ├── model/
│   │   ├── cifar10_linear_probe.py
│   │   ├── cifar10_linear_probe_sweep.py
│   │   ├── cifar10_zero_shot.py
│   │   ├── cifar10_zero_shot_batch.py
│   │   └── cifar10_zero_shot_batch_text_templates.py
│   ├── notebook/
│   │   └── CLIP_CIFAR10_Project.ipynb
│   │
    └── scripts/                      # Shell scripts for setting up environment and running Python scripts
│   │   ├── environment_setup.sh      # Script for setting up environment
│   │   └── python_scripts.sh         # Script to run evaluation scripts
├── original_openCLIP_repo/           # Folder containing files from original Open-CLIP repo
└── README.md                     # This file
```


## Installation

### Step 1: Install Dependencies and Setup Environment
Install the required Python packages using conda:

```bash
conda create -n open_clip python=3.8 -y
conda activate open_clip
pip install -r requirements.txt
```

### Step 2: Running the Project

#### Jupyter Notebook

Running the notebook (`CLIP_CIFAR10_Project.ipynb`) requires approximately **11GB of GPU memory**. To reduce GPU usage, you can modify the `batch_size` parameter in the code.

To run the notebook locally:

1. **Ensure Python is Installed**: Verify you have a Python interpreter (e.g., Python 3.x) installed and configured with the dependencies listed in [Installation](#installation).

2. **Select the Kernel**:
   - Open the notebook in your editor (e.g., VS Code or Jupyter).
   - In the top-right corner, click the kernel version (e.g., "Select Kernel").
   - Choose the Python environment with `open_clip_torch` installed (e.g., select `open_clip` as the kernel if set up).

3. **Run the Notebook**:
   - Click "Run All" to execute all cells and perform the zero-shot and linear probe evaluations on CIFAR-10.
   - Results will appear below each cell.

**Note**: Adjust `batch_size` in the notebook if you encounter GPU memory issues.

#### Python Scripts

To run the all Python scripts for the evaluation tasks, you can use the python_scripts.sh file as follows:

```bash
bash cifar10_evaluation/scripts/python_scripts.sh
```

The script runs the following tasks:

1.	**Zero-shot evaluation**:

```bash
CUDA_VISIBLE_DEVICES=0 python cifar10_evaluation/model/cifar10_zero_shot_batch.py
```

2.	**Linear probe evaluation**:
```bash
CUDA_VISIBLE_DEVICES=0 python cifar10_evaluation/model/cifar10_linear_probe.py
```

3.	**Linear probe hyperparameter sweep**:
```bash
CUDA_VISIBLE_DEVICES=0 python cifar10_evaluation/model/cifar10_linear_probe_sweep.py
```

4.	**Zero-shot evaluation with text templates**:
```bash
CUDA_VISIBLE_DEVICES=0 python cifar10_evaluation/model/cifar10_zero_shot_batch_text_templates.py
```

**Note**: CUDA_VISIBLE_DEVICES:
This sets which GPU to use. Change the value (e.g., 0, 1) based on your system’s GPU configuration.

## Results

The results from this project are summarized below:

1.	Zero-shot accuracy using the template "a photo of a {class}":
    - Top-1 accuracy: 93.65%
	- Top-5 accuracy: 99.83%
2.	Linear probe evaluation:
    - The best L2 regularization strength λ = 0.01 with a validation accuracy of 96.79%.
	- After retraining on the full training set (50,000 samples), the test accuracy is 96.79%.
3.	Zero-shot accuracy using the best template "a photo with the main subject of a {class}":
    - Top-1 accuracy: 94.14%

## Challenges and Improvements

Challenges:

1.	CPU Usage in Logistic Regression: The training process was slow due to the lack of GPU support in scikit-learn’s LogisticRegression. This was addressed by suggesting GPU-accelerated alternatives like the cuML library.
2.	Limited Text Template Variations: The CIFAR-10 dataset lacks detailed descriptions of the classes compared to datasets like Oxford-IIIT Pets or Food101. This was mitigated by experimenting with various text templates to improve model performance.

Potential Improvements:

GPU-Accelerated Logistic Regression: Switching to GPU-accelerated logistic regression using libraries such as cuML can speed up training.

## Citation

If you found this repository useful, please consider citing the following papers:

@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
