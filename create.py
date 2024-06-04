import shutil
import os

def copy_directories(start_sequence, end_sequence):
    base_source_path = "/mnt/ssd2/jihun/dataset/sequences/"
    base_destination_path = "/mnt/ssd2/jihun/dataset/multiframe/sequences/"

    start = int(start_sequence)
    end = int(end_sequence) + 1  # +1 because range is non-inclusive

    for seq_num in range(start, end):
        seq_folder = f"{seq_num:02}"  # Format number as two digits
        
        labels_source_path = os.path.join(base_source_path, seq_folder, "labels")
        velodyne_source_path = os.path.join(base_source_path, seq_folder, "velodyne")
        labels_destination_path = os.path.join(base_destination_path, seq_folder, "labels")
        velodyne_destination_path = os.path.join(base_destination_path, seq_folder, "velodyne")

        # Remove existing destination directory if it exists
        if os.path.exists(labels_destination_path):
            shutil.rmtree(labels_destination_path)
            print('removed {labels_destination_path}')
        if os.path.exists(velodyne_destination_path):
            shutil.rmtree(velodyne_destination_path)
            print('removed {velodyne_destination_path}')

        # Ensure the source directories exist before copying
        if os.path.exists(labels_source_path):
            shutil.copytree(labels_source_path, labels_destination_path)
        else:
            print(f"Labels directory not found for sequence {seq_folder}")

        if os.path.exists(velodyne_source_path):
            shutil.copytree(velodyne_source_path, velodyne_destination_path)
        else:
            print(f"Velodyne directory not found for sequence {seq_folder}")

# Function call
copy_directories("00", "10")
