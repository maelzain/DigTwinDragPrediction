import os
import logging
from PIL import Image
import torch
from torchvision import transforms

def load_and_preprocess_images(folder_path, resize=[64, 64], augment=False):
    """
    Loads all valid image files in a folder, converts them to grayscale,
    resizes them, applies optional augmentation, and converts to PyTorch tensors.
    
    Returns:
      - images_tensor: Stacked tensor of images.
      - orig_shapes: List of original image sizes.
      - filenames: List of image filenames.
    """
    image_list = []
    orig_shapes = []
    filenames = []
    # Compose transformation pipeline
    transform_list = [transforms.Resize(resize), transforms.ToTensor()]
    if augment:
        transform_list.insert(0, transforms.RandomHorizontalFlip())
    transform = transforms.Compose(transform_list)
    valid_exts = (".png", ".jpg", ".jpeg")
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
    for f in files:
        img_path = os.path.join(folder_path, f)
        try:
            img = Image.open(img_path).convert("L")
        except Exception as e:
            logging.error(f"Error opening image {img_path}: {e}")
            continue
        orig_shapes.append(img.size)
        img_tensor = transform(img)
        image_list.append(img_tensor)
        filenames.append(f)
        logging.info(f"Loaded image {f}: original size {img.size}, resized to {resize}")
    if image_list:
        images_tensor = torch.stack(image_list)
    else:
        images_tensor = torch.empty(0)
    return images_tensor, orig_shapes, filenames
