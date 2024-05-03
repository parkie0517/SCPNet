"""
    This file 'generate_multiframe.py' is used to create multiframe semantic KITTI dataset.
    Created by Heejun Park :)
"""
import numpy as np
import argparse
import os
import time


def count_files(directory):
    """
        Returns the number of files in the specified directory
    """
    full_path = os.path.abspath(directory) # get the full path name
    items = os.listdir(full_path) # get the list of files in the path
    # Filter out directories, count only files
    file_count = sum(os.path.isfile(os.path.join(full_path, item)) for item in items)
    
    return file_count

def get_data(file_base, dataset_path):
    """
        What does this function do?
            - read the data
            - return data
    """    

    
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

 
    def load_label_data(file_path):
        # Open the file in binary mode
        with open(file_path, 'rb') as file:
            # Read the file content into a numpy array of type uint16
            data = np.fromfile(file, dtype=np.uint16)
        
        return data.reshape((256, 256, 32))


    # 2. read the data
    bin_data = load_binary_data(bin_path)
    label_data = load_label_data(label_path)
    invalid_data = load_binary_data(invalid_path)
    occluded_data = load_binary_data(occluded_path)

    # 3. return the data
    return bin_data, label_data, invalid_data, occluded_data

def load_poses(poses_path):
    """
    1. pad index with zeros    
    2. read poses.txt file
    """
    with open(poses_path, 'r') as file:
        lines = file.readlines()
        poses = []
        for line in lines:
            # Split the line into floats
            numbers = np.array(list(map(float, line.strip().split())))
            # Reshape into a 3x4 matrix
            pose_matrix = numbers.reshape((3, 4))
            # convert to a 4x4 matrix
            pose_matrix_4x4 = np.vstack([pose_matrix, [0, 0, 0, 1]])
            poses.append(pose_matrix_4x4)

    return poses

def calculate_diff(pose1, pose2):
    """
    Input: 4x4 homogeneous matrices of i and j
    Output: rotational difference, translational difference

    1. Extract r and t
    2. calculate r diff
    3. calculate t diff
    """

    # 1. extract r and t
    r1, t1 = pose1[:3, :3], pose1[:3, 3]
    r2, t2 = pose2[:3, :3], pose2[:3, 3]

    # Calculate rotational difference
    rotational_diff = np.dot(np.linalg.inv(r1), r2)
    
    # Calculate translational difference
    translational_diff = t2 - t1

    return rotational_diff, translational_diff

def align_binary_data(data, rotation_diff, translation_diff):
    """
    this function is used to transform j to i

    1. transform
    2. filter (I use filter to only convert j-th voxels that will be inside the i-th coordinate frame)
    """
    aligned_data = np.zeros_like(data)

    # repeat this process for all the individual voxels (computationally heavy....)
    for z in range(data.shape[2]): # = range(0, 32)
        for y in range(data.shape[1]): # = range(0, 256)
            for x in range(data.shape[0]): # = range(0, 256)
                # 1. transform
                voxel_coords = np.array([x, y, z, 1])  # homogeneous coordinates
                rotated_coords = np.dot(rotation_diff, voxel_coords[:3])
                translated_coords = rotated_coords + translation_diff
                new_x, new_y, new_z = np.round(translated_coords).astype(int)

                # 2. filter
                if (0 <= new_x < data.shape[0] and 0 <= new_y < data.shape[1] and 0 <= new_z < data.shape[2]):
                    aligned_data[new_x, new_y, new_z] = data[x, y, z]

    return aligned_data


def align_label_data(data, rotation_diff, translation_diff):
    """
    this function is used to transform j label to i label
    * i made a distinct function just for transforming the "XXXXXX.label" file, because label file has an extra channel
    
    1. transform
    2. filter
    """
    aligned_data = np.zeros_like(data)
    
    # repeat for all the voxels
    for z in range(data.shape[2]):
        for y in range(data.shape[1]):
            for x in range(data.shape[0]):
                # 1. transformation
                voxel_coords = np.array([x, y, z, 1])  # homogeneous coordinates
                rotated_coords = np.dot(rotation_diff, voxel_coords[:3])
                translated_coords = rotated_coords + translation_diff

                new_x, new_y, new_z = np.round(translated_coords).astype(int)
                if (0 <= new_x < data.shape[0] and 0 <= new_y < data.shape[1] and 0 <= new_z < data.shape[2]):
                    for c in range(data.shape[3]):  # this for loops handles each class bit,   = range(0, 16)
                        aligned_data[new_x, new_y, new_z, c] = data[x, y, z, c]

    return aligned_data


def load_calib(calib_path):
    """
        1. read the calib.txt
        2. find the Tr data
        3. return it
    """
    # Initialize a list to hold the extracted numbers
    tr_numbers = []
    # Open and read the file line by line
    with open(calib_path, 'r') as file:
        for line in file:
            # Check if the line starts with 'Tr:'
            if line.startswith('Tr:'):
                # Split the line into parts after 'Tr:'
                parts = line.split()[1:]  # Skip the 'Tr:' part
                # Convert each part to a float and add to the list
                tr_numbers = [float(num) for num in parts]
                break
    matrix = np.array(tr_numbers).reshape(3, 4)
    return matrix


if __name__ == '__main__':
    start_time = time.time()

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
    output_dir = os.path.join(output, "voxels")
    if os.path.exists(output_dir):
        print("output directory already exists")
    else:
        os.makedirs(output_dir)
        print("output directory does not exist")
        print(f'{output_dir} path has been created')

    

    number_input_files = count_files(voxel_locaiton)

    number_distinct_input_files = int(number_input_files/4)  # bin, label, occluded, invalid. so I am dividing by 4
    
    sequence_length = (number_distinct_input_files-1)*increment

    # Variblaes needed for printing progress
    number_output_files = number_distinct_input_files - (n-1)
    progress_interval_percent = 10 # print every 10 percent
    progress_interval = number_output_files//progress_interval_percent 

    print(f'Location of dataset: {dataset}')
    print(f'Location of output directory: {output}')
    print(f'multiframe length: {n}')
    print(f'increment size: {increment}')
    print(f'files from 0 ~ {sequence_length} will be used')
    print(f'total number of output files: {number_output_files}')

    # 3. read poses.txt file (it's more efficient to read poses.txt just once)
    poses_location = os.path.join(dataset, "poses.txt")
    poses = load_poses(poses_location)

    # 4. read the calib.txt
    calib_location = os.path.join(dataset, "calib.txt")
    calibration = load_calib(calib_location)


    # Used for printing out the passed time during execution
    start_time = time.time()
    print("Begin Multi-frame Generation :)")

    # algorithm for creating the multi-frame semantic KITTI dataset
    for i in range(0, sequence_length - increment * (n-2), increment):
        # Create File Base String
        file_base = f"{i:06d}" # Convert i's data type from INT to STR and pad 0 at the front

        # read i-th data
        i_bin, i_label, i_invalid, i_occluded = get_data(file_base, dataset) # read i-th voxel data
        i_pose = poses[i] # read i-th pose


        
        for j in range(i + increment, i + increment * n, increment):
            # read j-th data
            j_bin, j_label, j_invalid, j_occluded = get_data(file_base, dataset) # read j-th voxel data
            j_pose = poses[j] # read j-th pose

            # now, let's calculate the pose difference between i and j
            rotational_diff, translation_diff = calculate_diff(i_pose, j_pose)


            # NOW WE SHALL BEGIN THE ALIGNING PROCESS! (align j into i-th space)
            # I need to optimize this code.... takes so faqing long
            aligned_j_bin = align_binary_data(j_bin, rotational_diff, translation_diff)
            aligned_j_label = align_label_data(j_label, rotational_diff, translation_diff)
            aligned_j_invalid = align_binary_data(j_invalid, rotational_diff, translation_diff)
            aligned_j_occluded = align_binary_data(j_occluded, rotational_diff, translation_diff)
  
            i_bin = add_binary_data(i_bin, aligned_j_bin)
            i_label = add_label_data(i_label, aligned_j_label)
            i_invalid = add_binary_data(i_invalid, aligned_j_invalid)
            i_occluded = add_binary_data(i_occluded, aligned_j_occluded)
        
        
        # Save fused scan
        print("Save Begin")
        np.packbits(i_bin).tofile(os.path.join(output_dir, f"{file_base}.bin"))
        # Save label file
        #i_label.tofile(os.path.join(output_dir, f"{file_base}.label"))  # Assuming i_label needs to be saved in 16-bit format
        i_label.astype(np.uint16).tofile(os.path.join(output_dir, f"{file_base}.label"))
        np.packbits(i_invalid).tofile(os.path.join(output_dir, f"{file_base}.invalid"))
        np.packbits(i_occluded).tofile(os.path.join(output_dir, f"{file_base}.occluded"))
        print("Save done")

        # Print progress & time
        if (i!= 0) and (i != progress_interval*increment*10) and (i % (progress_interval*increment) == 0.0):
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes_passed = int(elapsed_time / 60)
            seconds_passed = int(elapsed_time % 60)
            unge = sequence_length - increment * (n-1)
            print(f'Progress: {i/unge*100.0:.2f}%, Time Passes: {minutes_passed}:{seconds_passed:02d}')
    
    # Print final progress and time
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes_passed = int(elapsed_time / 60)
    seconds_passed = int(elapsed_time % 60)
    unge = sequence_length - increment * (n-1)
    print(f'Progress: 100.00%, Time Passes: {minutes_passed}:{seconds_passed:02d}')
    print("Multi-frame Generation Complete :D")