import torch
import hydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

sam2_ckpt = './dependencies/sam2ckpts/sam2.1_hiera_base_plus.pt'
# BUG: Use the sam2/configs/sam2.1/sam2.1_hiera_b+.yaml in the folder of site-packages
# not the relative path of the working directory
model_cfg = './configs/sam2.1/sam2.1_hiera_b+.yaml' 
predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_ckpt))

img = Image.open('r_10.png')
img = np.array(img,dtype=np.float32)/255.
# Convert to RGB with black background
# img = img.convert("RGB") 

# Convert to RGBA with white background
img = img[...,:3]*img[...,-1:] + (1. - img[...,-1:]) 

input_point = np.array([[500, 200]])
input_label = np.array([1.])


predictor.set_image(img)
masks, scores, logits = predictor.predict(point_coords=input_point,
                                              point_labels=input_label,
                                              multimask_output=True,
                                              )

sorted_ind = np.argsort(-scores) # Sort in descending order
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

from utils import *

show_masks(img, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
