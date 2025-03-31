import os
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from PIL import Image
from torchvision import transforms
from code.utils.csv_utils import normalize_drag_forces

class ReynoldsDataLoader:
    """
    Enhanced data loader for Reynolds experiments.
    Reads raw images and CSV files from each Reynolds folder,
    applies preprocessing (grayscale conversion, resizing, augmentation),
    and normalizes drag forces.
    """
    def __init__(self, config):
        self.config = config
        self.data_dir = config["data_dir"]
        self.reynolds = config["reynolds"]
        self.resize = config.get("resize", [64, 64])
        self.augment = config.get("augment", False)
        self.drag_normalization = config.get("drag_normalization", "minmax")
    
    def load_images(self, folder_path):
        image_list = []
        orig_shapes = []
        filenames = []
        valid_exts = (".png", ".jpg", ".jpeg")
        transform_list = [
            transforms.Resize(tuple(self.resize)),
            transforms.ToTensor()
        ]
        if self.augment:
            transform_list.insert(0, transforms.RandomHorizontalFlip())
        transform_list.append(transforms.Normalize(
            mean=[self.config.get("image_mean", 0.45)],
            std=[self.config.get("image_std", 0.22)]
        ))
        transform = transforms.Compose(transform_list)
        
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
        if not files:
            logging.error(f"No valid image files found in {folder_path}.")
        for f in files:
            img_path = os.path.join(folder_path, f)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("L")
                    orig_shapes.append(img.size)
                    img_tensor = transform(img)
                    image_list.append(img_tensor)
                    filenames.append(f)
            except Exception as e:
                logging.error(f"Error processing image {img_path}: {e}")
        if image_list:
            images_tensor = torch.stack(image_list)
        else:
            images_tensor = torch.empty(0)
        return images_tensor, orig_shapes, filenames

    def load_csv(self, csv_path):
        if not os.path.exists(csv_path):
            logging.error(f"CSV file not found: {csv_path}")
            return None
        try:
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded CSV file: {csv_path}, shape: {df.shape}")
            df = normalize_drag_forces(df, column_name="cd", method=self.drag_normalization)
            return df
        except Exception as e:
            logging.error(f"Error loading CSV file {csv_path}: {e}")
            return None

    def load_dataset(self):
        dataset = {}
        for re_val in self.reynolds:
            folder = f"re{re_val}"
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path):
                logging.warning(f"Folder {folder_path} does not exist. Skipping.")
                continue
            images, orig_shapes, filenames = self.load_images(folder_path)
            csv_filename = f"{folder}.csv"
            csv_path = os.path.join(folder_path, csv_filename)
            drag_df = self.load_csv(csv_path)
            if drag_df is not None:
                try:
                    drag_tensor = torch.tensor(drag_df["cd"].astype(float).values, dtype=torch.float32).unsqueeze(1)
                except Exception as e:
                    logging.error(f"Error converting drag data for folder {folder}: {e}")
                    drag_tensor = None
            else:
                drag_tensor = None

            if images is not None and images.numel() > 0 and drag_tensor is not None:
                if images.shape[0] != drag_tensor.shape[0]:
                    logging.warning(f"Size mismatch in folder {folder}: images={images.shape[0]} vs drag={drag_tensor.shape[0]}. Trimming to minimum count.")
                    min_count = min(images.shape[0], drag_tensor.shape[0])
                    images = images[:min_count]
                    drag_tensor = drag_tensor[:min_count]

            dataset[folder] = {
                "images": images,
                "original_shapes": orig_shapes,
                "drag": drag_tensor,
                "filenames": filenames
            }
            if images.numel() > 0 and drag_tensor is not None:
                logging.info(f"Folder {folder}: images shape {images.shape}, drag shape {drag_tensor.shape}")
        return dataset
