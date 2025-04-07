import torch
import os
from PIL import Image
import open_clip
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained CLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)

model.eval()  # Set the model to evaluation mode
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Download the CIFAR-10 test dataset
cifar10 = CIFAR10(root=os.path.expanduser("data"), download=True, train=False, transform=preprocess)

# Define multiple variations of text descriptions for CIFAR-10 classes
text_variations = [
    # Simple description
    ("Simple description", lambda c: f"a photo of a {c}"),      # 93.65%
    ("Image description", lambda c: f"an image of a {c}"),      # 93.62%

    # Detailed description
    # ("Detailed description", lambda c: f"a detailed photo of a {c} with fine details"),       92.94%
    ("Detailed description", lambda c: f"a photo with the main subject of a {c}"),              # 94.14%


    # Adding adjectives
    # ("Large description", lambda c: f"a large photo of a {c}"),       91.98%
    # ("Small description", lambda c: f"a small photo of a {c}"),       93.06%

    # Sentence structure variations
    ("Sentence structure 1", lambda c: f"this is a photo of a {c}"),        # 93.39%
    ("Sentence structure 2", lambda c: f"a beautiful photo of a {c}"),      # 93.66%
]

accuracies = {}

# Iterate over the different text input variations
for variation_name, description in text_variations:
    # Prepare text inputs
    text_inputs = torch.cat([tokenizer(description(c)) for c in cifar10.classes]).to(device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize

    correct_top1 = 0
    correct_top5 = 0
    total = len(cifar10)

    # Loop through all the test samples in batches
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(cifar10, batch_size=2048)):
            image_features = model.encode_image(images.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Get the top 5 predictions for each image in the batch
            values, indices = similarity.topk(5, dim=-1)

            # Calculate top-1 and top-5 accuracy for each image in the batch
            for i in range(len(labels)):
                # Check if the true class is in the top 1 prediction
                if indices[i, 0].item() == labels[i]:
                    correct_top1 += 1

                # Check if the true class is in the top 5 predictions
                if labels[i] in indices[i].tolist():
                    correct_top5 += 1

    # Calculate top-1 and top-5 accuracy
    top1_accuracy = correct_top1 / total * 100
    top5_accuracy = correct_top5 / total * 100

    # Store the results for this variation
    accuracies[variation_name] = (top1_accuracy, top5_accuracy)

# Output the accuracies for all variations
for variation_name, (top1, top5) in accuracies.items():
    print(f"Text variation: {variation_name}")
    print(f"  Top-1 accuracy: {top1:.2f}%")
    print(f"  Top-5 accuracy: {top5:.2f}%")