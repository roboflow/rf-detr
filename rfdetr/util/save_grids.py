import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from rfdetr.util.box_ops import box_cxcywh_to_xyxy
import torchvision.transforms as T

class DatasetGridSaver:
    """
    Utility class for saving images in grid. Allows visualization of the effects
    of augmentation on training and validation datasets on 3x3 grid of images
    
    Args:
        data_loader (DataLoader) : Dataloader of the dataset to display samples
        output_dir (Path) : Directory in which the images will be saved
        max_batches (int) : Number of batches to get the samples from
        dataset_type (str) : Type of dataset. 'train', 'val'
    """
    def __init__(self, data_loader : DataLoader, output_dir: Path, max_batches : int = 3, dataset_type : str = 'train'):
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.max_batches = max_batches
        self.dataset_type = dataset_type
        # Create the output_dir if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_grid(self):
        """
        Create and save the grid(s) inside output_dir
        """
        # Define the inverse transform to de-normalize images
        inv_normalize = T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
            )
        for batch_idx, (sample, target) in enumerate(self.data_loader):
            if batch_idx >= self.max_batches:
                break
            
            # Create a 3x3 grid for displaying images
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            fig.suptitle(f'{self.dataset_type} dataset, batch {batch_idx}')
            axes = axes.flatten()
            
            # Iterate through each image in the batch
            for sample_index, (single_image, single_target) in enumerate(zip(sample.tensors, target)):
                if sample_index >= 9:  # We only want to display the first 9 images in each batch
                    break

                resized_size = single_target['size']
                
                # Convert image tensor to numpy array for processing
                de_normalized_img = inv_normalize(single_image)
                img_numpy = (np.array(de_normalized_img).transpose(1, 2, 0)).copy()

                # Draw bounding boxes and labels on the image
                for (box, label) in zip(single_target['boxes'], single_target['labels']):
                    int_label = int(label)
                    
                    # Convert bounding box from cx,cy,wh format to xyxy
                    b = box_cxcywh_to_xyxy(box)
                    
                    # Scale bounding box coordinates to match the resized image
                    x_min, y_min, x_max, y_max = int(b[0] * resized_size[1]), int(b[1] * resized_size[0]),\
                                                int(b[2] * resized_size[1]), int(b[3] * resized_size[0])
                    
                    # Draw the bounding box on the image
                    cv2.rectangle(img_numpy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Add label text near the bounding box
                    text_size = cv2.getTextSize(str(int_label), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x, text_y = x_min, y_min - 10
                    cv2.rectangle(img_numpy, (text_x, text_y - text_size[1] - 5), 
                                (text_x + text_size[0] + 5, text_y + 5), (0, 255, 0), -1)  
                    cv2.putText(img_numpy, str(int_label), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                # Plot image in the grid
                ax = axes[sample_index]
                # Normalize image between 0.0 and 1.0 to show on matplotlib
                image = np.clip(img_numpy, 0.0, 1.0)
                ax.imshow(image)
                ax.axis('off')  # Hide axis
            # "Delete" empty axis
            for i in range(sample_index, 9):
              ax = axes[i]
              ax.axis('off')  # Hide axis
            # Adjust layout and save the figure
            fig.tight_layout()
            grid_path = self.output_dir / f"{self.dataset_type}_batch{batch_idx}_grid.jpg"
            plt.savefig(grid_path, dpi=200)
            plt.close()
            
        print(f"✅ Saved {self.dataset_type} grids with augmented images to: {self.output_dir.resolve()}")