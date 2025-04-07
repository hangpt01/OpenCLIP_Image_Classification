import torch
import os
from PIL import Image
import open_clip
from torchvision.datasets import CIFAR10


device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Download the dataset
cifar10 = CIFAR10(root=os.path.expanduser("data"), download=True, train=False)


# Prepare the inputs
image, class_id = cifar10[0]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([tokenizer(f"a photo of a {c}") for c in cifar10.classes]).to(device)


with torch.no_grad():
    # import pdb; pdb.set_trace()
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)
import pdb; pdb.set_trace()
top_1_accuracy = np.mean((test_labels == class_id).astype(float)) * 100.

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar10.classes[index]:>16s}: {100 * value.item():.2f}%")