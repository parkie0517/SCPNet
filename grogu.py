

import numpy as np



# Example usage:
file_path = './../dataset/sequences/00/voxels/000000.label'
voxel_labels = read_label_file(file_path)
print(type(voxel_labels[123, 123, 21]))
