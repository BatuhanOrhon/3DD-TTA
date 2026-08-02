import os
import numpy as np

def check_point_counts(data_root="./data/modelnet40_c"):
    corruptions = ['uniform', 'gaussian', 'background', 'impulse', 'upsampling',
                   'distortion_rbf', 'distortion_rbf_inv', 'density', 'density_inc',
                   'shear', 'rotation', 'cutout', 'distortion', 'occlusion', 'lidar']
    
    print(f"Checking point cloud counts in {data_root}...\n")
    
    for corruption in corruptions:
        filename = os.path.join(data_root, f"data_{corruption}_5.npy")
        
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            continue
            
        data = np.load(filename, allow_pickle=True)
        
        if isinstance(data, np.ndarray) and data.ndim == 3:
            # Shape is usually (num_samples, num_points, 3)
            num_samples, num_points, dims = data.shape
            print(f"[{corruption}] -> {num_samples} samples, EXACTLY {num_points} points per sample, {dims} dimensions.")
        else:
            # If it's an array of objects (lists of different lengths)
            point_counts = [len(pc) for pc in data]
            avg_points = np.mean(point_counts)
            min_points = np.min(point_counts)
            max_points = np.max(point_counts)
            print(f"[{corruption}] -> {len(data)} samples. Points per sample: Avg={avg_points:.1f}, Min={min_points}, Max={max_points}")

if __name__ == "__main__":
    check_point_counts()
