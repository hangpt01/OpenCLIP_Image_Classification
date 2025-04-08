# OpenCLIP: Zero-Shot and Linear Probe Evaluation of CLIP ViT-B-32 on CIFAR-10

This project is based on the CLIP (Contrastive Language-Image Pre-training) model introduced by Radford et al. [2021](https://arxiv.org/abs/2103.00020). We evaluate the **CLIP ViT-B-32** model for zero-shot image classification and linear probe classification on the **CIFAR-10** dataset.

## Objective
The main objectives of this project are:
1. **Zero-shot evaluation** using the CLIP ViT-B-32 model on CIFAR-10 with various text descriptions.
2. **Linear probe evaluation** after training the model using logistic regression with hyperparameter tuning.
3. **Exploring text template variations** to analyze how different textual descriptions affect zero-shot performance.

## Project Files

Here is the directory structure of the project:

OpenCLIP_Image_Classification/
│
├── .github/                      # GitHub-specific files
│
├── cifar10_evaluation/           # Folder containing evaluation scripts
│   ├── model/
│   │   ├── cifar10_linear_probe.py
│   │   ├── cifar10_linear_probe_sweep.py
│   │   ├── cifar10_zero_shot.py
│   │   ├── cifar10_zero_shot_batch.py
│   │   └── cifar10_zero_shot_batch_text_templates.py
│   └── notebook/
│       └── CLIP_CIFAR10_Project.ipynb
│
├── scripts/                      # Shell scripts for setting up environment and running Python scripts
│   ├── environment_setup.sh      # Script for setting up environment
│   └── python_scripts.sh         # Script to run evaluation scripts
└── README.md                     # This file

## Installation

Ensure you have **Python 3.x** installed. You can use `virtualenv` or `conda` for environment management.

### Step 1: Install Dependencies and Setup Environment
Install the required Python packages using:

```bash
pip install open_clip_torch
pip install ftfy regex tqdm
```

If you are using conda:
```bash
bash cifar10_evaluation/scripts/environment_setup.sh
```

### Step 2: Running the Project

#### Jupyter Notebook

To run the Jupyter notebook (CLIP_CIFAR10_Project.ipynb), simply execute:

```bash
jupyter notebook cifar10_evaluation/notebook/CLIP_CIFAR10_Project.ipynb
```

This will open the notebook in your browser where you can interact with the zero-shot and linear probe evaluations on CIFAR-10.

#### Python Scripts

To run the various Python scripts for the evaluation tasks, you can use the python_scripts.sh file as follows:

```bash
bash cifar10_evaluation/scripts/python_scripts.sh
```

The script runs the following tasks:

1.	Zero-shot evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python cifar10_evaluation/model/cifar10_zero_shot_batch.py
```

	2.	Linear probe evaluation:
```bash
CUDA_VISIBLE_DEVICES=0 python cifar10_evaluation/model/cifar10_linear_probe.py
```

	3.	Linear probe hyperparameter sweep:
```bash
CUDA_VISIBLE_DEVICES=0 python cifar10_evaluation/model/cifar10_linear_probe_sweep.py
```

	4.	Zero-shot evaluation with batch and text templates:
```bash
CUDA_VISIBLE_DEVICES=0 python cifar10_evaluation/model/cifar10_zero_shot_batch_text_templates.py
```


CUDA_VISIBLE_DEVICES:
This sets which GPU to use. Change the value (e.g., 0, 1) based on your system’s GPU configuration.

## Results

The results from this project are summarized below:
	1.	Zero-shot accuracy using the template "a photo of a {class}":
	•	Top-1 accuracy: 93.65%
	•	Top-5 accuracy: 99.83%
	2.	Linear probe evaluation:
	•	The best L2 regularization strength (\lambda = 0.01) with a validation accuracy of 96.79%.
	•	After retraining on the full training set (50,000 samples), the test accuracy is 96.79%.

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
