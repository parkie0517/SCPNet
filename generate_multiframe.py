import numpy as np
import argparse
# from pyquaternion import Quaternion

def get_data(index):
    # Placeholder for your data loading logic
    # Should return data arrays for bin, label, invalid, and occluded
    return None, None, None, None

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
    n = args.number
    increment = args.increment
    

    # this should be automatically done
    sequence_length = 4540  # Adjust based on your dataset specifics

    print(dataset)
    print(n)
    print(increment)
    print(sequence_length)


    for i in range(0, sequence_length - increment * (n-1), increment):
        i_bin, i_label, i_invalid, i_occluded = get_data(i)
        i_pose = get_pose(i)

        for j in range(i + increment, i + increment * n, increment):
            j_bin, j_label, j_invalid, j_occluded = get_data(j)
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
          