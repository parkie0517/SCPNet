import numpy as np
import argparse
import os

def count_files(directory):
    """
    Returns the number of files in the specified directory
    """
    full_path = os.path.abspath(directory) # get the full path name
    items = os.listdir(full_path) # get the list of files in the path
    # Filter out directories, count only files
    file_count = sum(os.path.isfile(os.path.join(full_path, item)) for item in items)
    
    return file_count

def get_data(index, dataset_path):
    """
    1. pad index with zeros
    2. read the data
    3. return data
    """    
    # 1. pad index with zeros
    file_base = f"{index:06d}"
    
    # build file path
    bin_path = os.path.join(dataset_path, "voxels", f"{file_base}.bin")
    label_path = os.path.join(dataset_path, "voxels", f"{file_base}.label")
    invalid_path = os.path.join(dataset_path, "voxels", f"{file_base}.invalid")
    occluded_path = os.path.join(dataset_path, "voxels", f"{file_base}.occluded")

    # Function to read and reshape binary data
    def load_binary_data(file_path):
        with open(file_path, 'rb') as file:
            data = np.fromfile(file, dtype=np.uint8)  # Read data as 8-bit unsigned integers
            bits = np.unpackbits(data)  # Convert bytes to bits
            return bits.reshape((256, 256, 32))  # Reshape to 3D array


    # Function to read label data
    def load_label_data(file_path):
        with open(file_path, 'rb') as file:
            data = np.fromfile(file, dtype=np.uint8)  # Read data as 16-bit unsigned integers
            bits = np.unpackbits(data)  # Convert bytes to bits
            return bits.reshape((256, 256, 32, 16))  # Reshape to 3D array


    # 2. read the data
    bin_data = load_binary_data(bin_path)
    label_data = load_label_data(label_path)
    invalid_data = load_binary_data(invalid_path)
    occluded_data = load_binary_data(occluded_path)

    # 3. return the data
    return bin_data, label_data, invalid_data, occluded_data

def get_pose(index):
    # Placeholder for pose loading logic
    # Should return a pose matrix or similar data structure
    return None

def calculate_diff(pose1, pose2):
    # Placeholder to calculate rotational and translation differences between two poses
    # Should return rotation and translation differences
    return None, None

def align(data, rotation_diff, translation_diff):
    # Placeholder to align data based on rotation and translation differences
    # Should return aligned data
    return data

def filter(target_space, *data):
    # Placeholder to filter out data that do not fit in the target space
    # Should return filtered data
    return data

def add(*data):
    # Placeholder to add data together
    # Should return combined data
    return i_bin, i_label, i_invalid, i_occluded



if __name__ == '__main__':
    # 1. argument settings
    parser = argparse.ArgumentParser(
        description='code for generating multiframe semantic-KITTI dataset for semantic scene completion task'
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='should be like "..../dataset/sequences/00',
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='type in the output directory',
    )

    parser.add_argument(
        '--number', '-n',
        default='4',
        type=int,
        required=False,
        help='number of frames used to create the multiframe data',
    )

    parser.add_argument(
        '--increment', '-i',
        default=5,
        type=int,
        required=False,
        help='increment size. default is 5',
    )

    args = parser.parse_args() # returns the arguments provided by the user or the default
    dataset = args.dataset
    output = args.output
    n = args.number
    increment = args.increment
    

    # this should be automatically done
    voxel_locaiton = os.path.join(dataset, "voxels/")

    # 2. output directory settings
    if os.path.exists(output):
        print("output directory already exists")
    else:
        os.makedirs(output)
        print("output directory does not exist")
        print(f'{output} path has been created')

    number_files = count_files(voxel_locaiton)
    number_files = int((number_files/n) * increment)  # Adjust based on your dataset specifics

    sequence_length = number_files - increment
    print(f'Location of dataset: {dataset}')
    print(f'Location of output directory: {output}')
    print(f'number of multiframe: {n}')
    print(f'increment size: {increment}')
    print(f'files from 0 ~ {sequence_length} will be used')


    # 3. algorithm for creating the multi-frame semantic KITTI dataset
    for i in range(0, sequence_length - increment * (n-1), increment):
        i_bin, i_label, i_invalid, i_occluded = get_data(i, dataset)
        i_pose = get_pose(i)
        exit(0)
        for j in range(i + increment, i + increment * n, increment):
            j_bin, j_label, j_invalid, j_occluded = get_data(j, dataset)
            j_pose = get_pose(j)

            rotational_diff, translation_diff = calculate_diff(i_pose, j_pose)
            aligned_j_bin, aligned_j_label, aligned_j_invalid, aligned_j_occluded = align(
                (j_bin, j_label, j_invalid, j_occluded), rotational_diff, translation_diff
            )

            filtered_j_bin, filtered_j_label, filtered_j_invalid, filtered_j_occluded = filter(
                i_bin, aligned_j_bin, aligned_j_label, aligned_j_invalid, aligned_j_occluded
            )

            i_bin, i_label, i_invalid, i_occluded = add(
                i_bin, i_label, i_invalid, i_occluded, filtered_j_bin, filtered_j_label, filtered_j_invalid, filtered_j_occluded
            )
        """    
        if i % 100 == 0:
            print(f'file {i} done')
        """
          