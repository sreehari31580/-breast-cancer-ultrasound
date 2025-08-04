import cv2
import numpy as np
from glob import glob
from PIL import Image

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    return img

def register_image_to_atlas(img, atlas_img):
    # ECC registration (affine)
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    try:
        cc, warp_matrix = cv2.findTransformECC(atlas_img, img, warp_matrix, warp_mode, criteria)
        aligned = cv2.warpAffine(img, warp_matrix, (224, 224), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except cv2.error:
        aligned = img
    return aligned

def multi_atlas_registration(img_path, atlas_paths):
    img = read_image(img_path)
    registered_imgs = []
    for atlas_path in atlas_paths:
        atlas_img = read_image(atlas_path)
        reg_img = register_image_to_atlas(img, atlas_img)
        registered_imgs.append(reg_img)
    return registered_imgs  # list of registered images, one per atlas