import torch
import os
from PIL import Image
import open_clip
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', device=device)
model.eval()  # Set model in evaluation mode to disable training-specific behaviors
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Download the CIFAR-10 dataset
root = os.path.expanduser("data")
train = CIFAR10(root, download=True, train=True, transform=preprocess)
test = CIFAR10(root, download=True, train=False, transform=preprocess)

# Function to move data and compute features on the GPU
def get_features(dataset):
    all_features = []
    all_labels = []

    # Efficient data transfer with pinned memory and no gradient computation
    data_loader = DataLoader(dataset, batch_size=2048, pin_memory=True, num_workers=4)

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            # Move images to GPU before passing them to the model
            images = images.to(device, non_blocking=True)
            features = model.encode_image(images)

            all_features.append(features)
            all_labels.append(labels)

    # Return features and labels as numpy arrays
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Compute features for both training and testing datasets
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Train Logistic Regression Classifier
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate the classifier's accuracy
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")