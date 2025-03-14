import os
import logging
import torch
from code.utils.image_utils import load_and_preprocess_images
from code.utils.csv_utils import preprocess_drag_csv

class ReynoldsDataLoader:
    def __init__(self, config):
        self.data_dir = config["data_dir"]
        self.reynolds = config["reynolds"]
        self.resize = config.get("resize", [64, 64])
        self.augment = config.get("augment", False)
        self.drag_normalization = config.get("drag_normalization", "standard")

    def load_dataset(self):
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

            # Ensure matching number of images and drag samples
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
                logging.info(f"Folder {folder}: images={images.shape}, drag={drag_tensor.shape}")
        return dataset

    def extract_times(self, filenames):
        times = []
        for name in filenames:
            try:
                # Example: "timestep_10.png" -> "timestep_10" -> "10"
                base = os.path.splitext(name)[0]
                parts = base.split("_")
                time_val = float(parts[1])
                times.append(time_val)
            except Exception as e:
                logging.error(f"Error extracting time from filename {name}: {e}")
                continue
        return times if times else None
