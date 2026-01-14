import os
import torch
import numpy as np
import cv2
import json
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_single_torch_file(file_path, output_dir):
    """
    Process a single .torch file and save images and poses.
    
    Args:
        file_path (str): Path to the specific .torch file
        output_dir (str): Base directory to save outputs
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return False

        # Create output directories
        images_dir = os.path.join(output_dir, 'images')
        meta_dir = os.path.join(output_dir, 'metadata')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        
        logging.info(f"Loading file: {file_path}")
        # Load the torch file
        data = torch.load(file_path)
        
        logging.info(f"Processing {len(data)} scenes in the file...")

        # Process each scene in the file
        for cur_scene in data:
            # Get the key from the first element to use as subdirectory name
            scene_name = cur_scene['key']
            if isinstance(scene_name, torch.Tensor):
                scene_name = scene_name.item()
            
            # Create subdirectories for this specific sequence
            seq_images_dir = os.path.join(images_dir, str(scene_name))
            os.makedirs(seq_images_dir, exist_ok=True)
            
            cur_info_dict = {
                'scene_name': scene_name,
                'frames': []
            }
            
            cur_pose_info = cur_scene['cameras']
            
            # Pre-allocate lists for better memory efficiency
            frames = []
            
            # Process each element in the list
            for img_idx, img_data in enumerate(cur_scene['images']):
                try:
                    # Convert tensor to numpy if needed
                    if isinstance(img_data, torch.Tensor):
                        img_data = img_data.numpy()
                    
                    # Convert PIL image to numpy array more efficiently
                    img_array = np.frombuffer(img_data.tobytes(), dtype=np.uint8)
                    img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img_array is None:
                        raise ValueError("Failed to decode image data")
                    
                    h, w = img_array.shape[:2]
                    
                    # Save as PNG using cv2
                    img_path = os.path.join(seq_images_dir, f'{img_idx:05d}.png')
                    if not cv2.imwrite(img_path, img_array):
                        raise ValueError(f"Failed to write image to {img_path}")
                    
                    # Convert pose info tensors to regular Python types if needed
                    pose_data = cur_pose_info[img_idx]
                    if isinstance(pose_data, torch.Tensor):
                        pose_data = pose_data.tolist()
                    
                    # Calculate camera parameters
                    fx, fy, cx, cy = map(float, [
                        pose_data[0] * w,
                        pose_data[1] * h,
                        pose_data[2] * w,
                        pose_data[3] * h
                    ])
                    
                    # Calculate world to camera transform
                    w2c = np.array(pose_data[6:], dtype=np.float32).reshape(3, 4)
                    w2c = np.vstack([w2c, [0, 0, 0, 1]])
                    
                    frame_info = {
                        'image_path': img_path,  # Absolute path or relative to output_dir
                        'fxfycxcy': [fx, fy, cx, cy],
                        'w2c': w2c.tolist()
                    }
                    frames.append(frame_info)
                    
                except Exception as e:
                    logging.error(f"Error processing image {img_idx} in {file_path}: {str(e)}")
                    continue
            
            cur_info_dict['frames'] = frames
            
            # Save metadata
            meta_path = os.path.join(meta_dir, f'{scene_name}.json')
            with open(meta_path, 'w') as f:
                json.dump(cur_info_dict, f, indent=4)
            
            logging.info(f"Saved scene {scene_name} to {meta_path}")
                
        return True

    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single .torch file for LVSM")
    
    # Changed arguments to accept a single file path
    parser.add_argument("--file_path", type=str, required=True, help="Path to the specific .torch file to process")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed images and metadata")
    
    args = parser.parse_args()
    
    logging.info("Starting single file processing...")
    
    success = process_single_torch_file(args.file_path, args.output_dir)
    
    if success:
        logging.info("Processing completed successfully!")
    else:
        logging.info("Processing failed.")