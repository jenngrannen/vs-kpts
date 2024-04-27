import torch
import random
import cv2
from typing import Callable, List, Tuple
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import pickle
import os
#from transformers import AutoModel, AutoTokenizer
from datetime import datetime
#import imgaug.augmenters as iaa

# No domain randomization
transform = transforms.Compose([transforms.ToTensor()])

# Domain randomization
#transform = transforms.Compose([
#    iaa.Sequential([
#        iaa.AddToHueAndSaturation((-20, 20)),
#        iaa.LinearContrast((0.85, 1.2), per_channel=0.25),
#        iaa.Add((-10, 30), per_channel=True),
#        iaa.GammaContrast((0.85, 1.2)),
#        iaa.GaussianBlur(sigma=(0.0, 0.6)),
#        iaa.ChangeColorTemperature((5000,35000)),
#        iaa.MultiplySaturation((0.95, 1.05)),
#        iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
#    ], random_order=True).augment_image,
#    transforms.ToTensor()
#])

def normalize(x):
    return F.normalize(x, p=1)

def gauss_2d_batch(width, height, sigma, kpt, normalize_dist=False):
    U, V = kpt
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    if normalize_dist:
        return normalize(G).double()
    G = torch.unsqueeze(G, 0)
    return G.double()

def vis_gauss(gaussians):
    gaussians = gaussians.cpu().numpy()
    h1,h2,h3,h4 = gaussians
    output = cv2.normalize(h1, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test.png', output)

class LanguageKeypointsDataset(Dataset):
    def __init__(self, num_keypoints, data_dir, img_height, img_width, transform, gauss_sigma=8):
        self.num_keypoints = num_keypoints
        self.img_height = img_height
        self.img_width = img_width
        self.gauss_sigma = gauss_sigma
        self.transform = transform

        self.data_dir = data_dir

        # Set Embedding to None, Initially
        self.embedding, self.exemplar = None, None

        with open(os.path.join(self.data_dir, "annots.pkl"), 'rb') as f:
            self.full_annots = pickle.load(f)

        #self.split_annot_keys = np.array(list(self.full_annots.keys()))[list(indices)]
        self.split_annot_keys = list(self.full_annots.keys())

        self.rgb_paths = []
        self.lang_ref = []
        self.kpt_annots = []

        for img_k in self.split_annot_keys:
            img_name = os.path.join(self.data_dir, img_k)
            img_langs = self.full_annots[img_k].keys()
            for lang_k in img_langs:
                self.rgb_paths.append(img_name)
                self.lang_ref.append(lang_k)
                self.kpt_annots.append(self.full_annots[img_k][lang_k][0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_raw = cv2.imread(self.rgb_paths[idx])
        rgb_final = self.transform(rgb_raw)

        # some lang stuff here
        # lang_emb = self.embed()
        # print("lang_emb", lang_emb.shape)

        kpt_scaled = np.array(self.kpt_annots[idx])
        gauss_kpt = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, torch.from_numpy(kpt_scaled))

        return rgb_final, self.lang_ref[idx], gauss_kpt, rgb_raw

    def __len__(self):
        return len(self.rgb_paths)

if __name__ == '__main__':
    NUM_KEYPOINTS = 1
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    GAUSS_SIGMA = 10
    TEST_DIR = ""
    test_dataset = KeypointsDataset('/host/data/%s/test/images'%TEST_DIR,
                         NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, transform, gauss_sigma=GAUSS_SIGMA)
    img, gaussians = test_dataset[0]
    vis_gauss(gaussians)
