import os
import logging
import torch
from torch.utils.data import TensorDataset
from code.utils.image_utils import load_and_preprocess_images
from code.utils.csv_utils import preprocess_drag_csv

class ReynoldsDataLoader:
    """
    Loads and preprocesses the dataset for multiple Reynolds experiments.
    Combines images and corresponding drag CSV data.
    """
    def __init__(self, config):
        self.data_dir = config["data_dir"]
        self.reynolds = config["reynolds"]
        self.resize = config.get("resize", [64, 64])
        self.augment = config.get("augment", False)
        self.drag_normalization = config.get("drag_normalization", "minmax")

    def load_dataset(self):
        """
        Iterates through each Reynolds folder, loads images and drag data,
        and returns a dictionary with keys corresponding to each folder.
        """
        dataset = {}
        for re_val in self.reynolds:
            folder = f"re{re_val}"
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path):
                logging.warning(f"Folder {folder_path} does not exist. Skipping.")
                continue
            images, orig_shapes, filenames = load_and_preprocess_images(
                folder_path, resize=self.resize, augment=self.augment
            )
            csv_filename = f"{folder}.csv"
            csv_path = os.path.join(folder_path, csv_filename)
            drag_data = preprocess_drag_csv(csv_path, column_name="cd", norm_method=self.drag_normalization)
            if drag_data is not None:
                try:
                    drag_tensor = torch.tensor(
                        drag_data["cd"].astype(float).values, dtype=torch.float32
                    ).unsqueeze(1)
                except Exception as e:
                    logging.error(f"Error converting drag data for folder {folder}: {e}")
                    drag_tensor = None
            else:
                drag_tensor = None

            # Ensure the number of images and drag values match
            if images is not None and images.numel() > 0 and drag_tensor is not None:
                if images.shape[0] != drag_tensor.shape[0]:
                    logging.warning(
                        f"Size mismatch in folder {folder}: images={images.shape[0]} vs drag={drag_tensor.shape[0]}. Trimming to min count."
                    )
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

    def extract_times(self, filenames):
        """
        Extracts time values from filenames that follow the format "timestep_XXXX.png".
        Returns a list of float time values.
        """
        times = []
        for name in filenames:
            try:
                base = os.path.splitext(name)[0]
                parts = base.split("_")
                time_val = float(parts[1])
                times.append(time_val)
            except Exception as e:
                logging.error(f"Error extracting time from filename {name}: {e}")
                continue
        return times if times else None
