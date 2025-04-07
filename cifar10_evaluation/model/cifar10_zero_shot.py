import torch
import os
from PIL import Image
import open_clip
from torchvision.datasets import CIFAR10
from tqdm import tqdm  # For progress bar

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained CLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
model.eval()  # Set the model to evaluation mode
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Download the CIFAR-10 test dataset
cifar10 = CIFAR10(root=os.path.expanduser("data"), download=True, train=False)

# Prepare text inputs (text descriptions for the CIFAR-10 classes)
text_inputs = torch.cat([tokenizer(f"a photo of a {c}") for c in cifar10.classes]).to(device)

# Initialize counters for accuracy calculation
correct_top1 = 0
correct_top5 = 0
total = len(cifar10)

# Loop through all the test samples
for i in tqdm(range(total)):
    # Prepare the image
    image, class_id = cifar10[i]
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Encode the image and text features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize
        text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize

    # Compute similarity between image and text features
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Get the top 5 predictions
    values, indices = similarity[0].topk(5)  # Get top 5 predictions

    # Check if the true class is in the top 1 prediction
    if indices[0].item() == class_id:
        correct_top1 += 1

    # Check if the true class is in the top 5 predictions
    if class_id in indices.tolist():
        correct_top5 += 1

# Calculate top-1 and top-5 accuracy
top1_accuracy = correct_top1 / total * 100
top5_accuracy = correct_top5 / total * 100

print(f"Top-1 accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 accuracy: {top5_accuracy:.2f}%")

'''
Top-1 accuracy: 93.65%
Top-5 accuracy: 99.83%
'''