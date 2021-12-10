import json
import os
import torch.nn.functional as F

from torchvision import models
import torch
from torchvision.transforms import transforms
from PIL import Image

IMG_DIR = 'images'
LABELS_PATH = 'imagenet-labels.json'


def load_labels(filename: str):
    with open(filename) as f:
        pet = json.loads(f.read())
    return pet


def get_batch(img_dir: str) -> tuple[torch.Tensor, list[str]]:
    img_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
    images = [Image.open(path) for path in img_paths]
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    images = [transform(image) for image in images]
    batch = torch.zeros(size=(len(images), 3, 224, 224))
    for i in range(len(images)):
        batch[i] = images[i]
    return batch, img_paths


def predict(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vgg16_model = models.vgg16(pretrained=True)
    vgg16_model.eval()
    with torch.no_grad():
        output = vgg16_model(batch).data
        _, index = torch.max(output, 1)
        percentage = (F.softmax(output, dim=1)[0] * 100)[index]
    return index, percentage


def main():
    images_to_predict, path_to_images = get_batch(IMG_DIR)
    class_idx, percentage = predict(images_to_predict)
    labels_dictionary = load_labels(LABELS_PATH)
    for i in range(len(class_idx)):
        print('image: ', path_to_images[i])
        print('predicted_class: ', labels_dictionary[str(int(class_idx[i]))])
        print('confidence ', percentage[i], '%')


if __name__ == '__main__':
    main()
