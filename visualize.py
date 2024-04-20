import os
import time
import argparse
import sys
import numpy as np
import torch
from tqdm import tqdm
from ply_writer import *
# from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name,get_eval_mask,unpack
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from utils.np_ioueval import iouEval
from utils.load_save_util import load_checkpoint
import glob
import warnings
import os
warnings.filterwarnings("ignore")
import yaml

color_map_reduced = [[0, 0, 0],[245, 150, 100],[245, 230, 100],[150, 60, 30],[180, 30, 80],[255, 0, 0],[30, 30, 255],[200, 40, 255],[90, 30, 150],[255, 0, 255],
                     [255, 150, 255],[75, 0, 75],[75, 0, 175],[0, 200, 255],[50, 120, 255],[0, 175, 0],[0, 60, 135],[80, 240, 150],[150, 240, 255],[0, 0, 255]]

def train2SemKITTI(input_label):
    # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
    return input_label + 1
def unpack(compressed):  # from samantickitti api
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def label2voxel(prediction_dir,original_dataset):
    predictions=glob.glob(f"{prediction_dir}/*.label")
    
    output_dir=prediction_dir + '/vis'
    # import pdb;pdb.set_trace()
    os.makedirs(output_dir, exist_ok=True)
    for pred in predictions:
        pred_vox = np.fromfile(pred, dtype=np.uint16)
        # color_map[pred_vox]
        pred_vox = remap_lut[pred_vox]
        
        invalid_name=os.path.join(original_dataset,'sequences','08','voxels',pred.split('/')[-1]).replace('.label','.invalid')
        invalid_voxels = unpack(np.fromfile(invalid_name, dtype=np.uint8))
        label_name=invalid_name.replace('invalid','label')
        labels = np.fromfile(label_name, dtype=np.uint16)
        labels = remap_lut[labels]
        masks = get_eval_mask(labels, invalid_voxels)
        masks=np.where(pred_vox==0,False,masks)
        pred_vox=pred_vox[masks]
        vox_idx=np.indices((256, 256, 32)).transpose((1, 2, 3, 0))
        vox_idx=vox_idx.reshape(-1,3)
        vox_idx=vox_idx[masks]
        colors=np.array(color_map_reduced)[pred_vox]
        write_ply(f"{output_dir}/{pred.split('/')[-1].replace('label','ply')}", [vox_idx.astype(np.uint8),colors.astype(np.uint8)], ['x', 'y', 'z','red', 'green', 'blue'])
        # import pdb;pdb.set_trace()
        
with open("config/label_mapping/semantic-kitti.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
class_remap = semkittiyaml["learning_map"]
maxkey2 = max(class_remap.keys())
remap_lut = np.zeros((maxkey2 + 100), dtype=np.int32)
remap_lut[list(class_remap.keys())] = list(class_remap.values())
remap_lut[remap_lut == 0] = 255   # map 0 to 'invalid'
remap_lut[0] = 0  # only 'empty' stays 'empty'.

label2voxel('/mnt/jihun5/tta-SCPNetv1/out_scpnet/val/sequences/08/predictions_s_5_lr_0.001_scanwise_iter5_strong_baseline','/mnt/jihun5/tta-SCPNetv2/dataset')