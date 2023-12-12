import os
from PIL import Image
import torch

from torchmetrics.image.kid import KernelInceptionDistance
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.PILToTensor()
])

# training_path = 'Dafu'
# generated_path = 'dreambooth_image'

training_path = 'pokemon_training'
generated_path = 'pokemon_generation'


# def get_images(path):
#     images = []
#     for file_path in os.listdir(path):
#         try:
#             image_path = os.path.join(path, file_path)
#             img = Image.open(image_path).resize((512, 512))
#             images.append(transform(img))
#         except:
#             print(f"{image_path} is not a valid image, please make sure to remove this file from the directory otherwise the training could fail.")
#     return torch.stack(images)

def get_images_pokemon(path,ends=None):
    images = []
    for file_path in os.listdir(path):
        try:
            image_path = os.path.join(path, file_path)
            if ends is None or image_path.endswith(ends):
                img = Image.open(image_path).resize((512, 512))
                images.append(transform(img))
        except:
            print(f"{image_path} is not a valid image, please make sure to remove this file from the directory otherwise the training could fail.")
    print(len(images))
    return torch.stack(images)

training_images = get_images_pokemon(training_path)
generated_images = get_images_pokemon(generated_path, '1.png')
metric = KernelInceptionDistance(subsets=10, subset_size=4)
metric.update(training_images, real=True)
metric.update(generated_images, real=False)
print(metric.compute())

