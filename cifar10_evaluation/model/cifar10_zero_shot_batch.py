import torch
import os
from PIL import Image
import open_clip
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bar

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained CLIP model
# https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
# ['openai', 'laion400m_e31', 'laion400m_e32', 'laion2b_e16', 'laion2b_s34b_b79k', 
# 'datacomp_xl_s13b_b90k', 'datacomp_m_s128m_b4k', 
# 'commonpool_m_clip_s128m_b4k', 'commonpool_m_laion_s128m_b4k', 'commonpool_m_image_s128m_b4k', 
# 'commonpool_m_text_s128m_b4k', 'commonpool_m_basic_s128m_b4k', 'commonpool_m_s128m_b4k', 
# 'datacomp_s_s13m_b4k', 'commonpool_s_clip_s13m_b4k', 'commonpool_s_laion_s13m_b4k', 'commonpool_s_image_s13m_b4k', 
# 'commonpool_s_text_s13m_b4k', 'commonpool_s_basic_s13m_b4k', 'commonpool_s_s13m_b4k',
#  'metaclip_400m', 'metaclip_fullcc']


model.eval()  # Set the model to evaluation mode
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Download the CIFAR-10 test dataset
cifar10 = CIFAR10(root=os.path.expanduser("data"), download=True, train=False, transform=preprocess)

# Prepare text inputs (text descriptions for the CIFAR-10 classes)
text_inputs = torch.cat([tokenizer(f"a photo of a {c}") for c in cifar10.classes]).to(device)
text_features = model.encode_text(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize


correct_top1 = 0
correct_top5 = 0
total = len(cifar10)

# Loop through all the test samples in batches
with torch.no_grad():
    for images, labels in tqdm(DataLoader(cifar10, batch_size=1024)):
        image_features = model.encode_image(images.to(device))
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get the top 5 predictions for each image in the batch
        values, indices = similarity.topk(5, dim=-1)  # Get top 5 predictions for all images in the batch

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

print(f"Top-1 accuracy: {top1_accuracy:.2f}%")
print(f"Top-5 accuracy: {top5_accuracy:.2f}%")

'''
laion2b_s34b_b79k
Top-1 accuracy: 93.65%
Top-5 accuracy: 99.83%

openai
Top-1 accuracy: 86.16%
Top-5 accuracy: 99.16%
'''