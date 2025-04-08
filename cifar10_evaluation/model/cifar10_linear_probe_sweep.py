import torch
import os
from PIL import Image
import open_clip
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
model.eval()  # Model in eval mode
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Download CIFAR-10 dataset
root = os.path.expanduser("data")
train = CIFAR10(root, download=True, train=True, transform=preprocess)
test = CIFAR10(root, download=True, train=False, transform=preprocess)

# Split the training dataset into 80% train and 20% validation
train_size = int(0.8 * len(train))  # 80% for training
val_size = len(train) - train_size  # 20% for validation
train_data, val_data = random_split(train, [train_size, val_size])
batch_size = 2048

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=batch_size)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate features for train, validation, and test sets
train_features, train_labels = get_features(train_data)
val_features, val_labels = get_features(val_data)
test_features, test_labels = get_features(test)

# Define hyperparameter sweep for regularization strength C (L2 regularization)
C_values = [10**-6, 10**-4, 10**-2, 1, 10**2, 10**4, 10**6]  # Logarithmic sweep

best_C = None
best_accuracy = 0

# Perform hyperparameter sweep over C
for C in C_values:
    print(f"Training with C = {C}")
    
    # Initialize the Logistic Regression classifier
    classifier = LogisticRegression(random_state=0, C=C, max_iter=1000, verbose=0)
    
    # Train the classifier on the training features
    classifier.fit(train_features, train_labels)
    
    # Evaluate the classifier on the validation set
    val_predictions = classifier.predict(val_features)
    val_accuracy = np.mean((val_labels == val_predictions).astype(float)) * 100.
    print(f"Validation Accuracy = {val_accuracy:.3f}%")
    
    # Track the best C value based on validation accuracy
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_C = C

print(f"Best C value: {best_C} with validation accuracy of {best_accuracy:.3f}%")

# Train the classifier on the entire training dataset with the best C
final_classifier = LogisticRegression(random_state=0, C=best_C, max_iter=1000, verbose=1)
final_classifier.fit(train_features, train_labels)

# Evaluate on the test set
test_predictions = final_classifier.predict(test_features)
test_accuracy = np.mean((test_labels == test_predictions).astype(float)) * 100.
print(f"Test Accuracy = {test_accuracy:.3f}%")