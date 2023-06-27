from torch.utils.data import Dataset
from os import listdir
from os.path import splitext, join
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import cv2
import logging
import random


def findTestKeypoints(image): 
    size = image.shape
    kpts = np.zeros((size[0]*size[1], 2, 1), dtype="float")
    count = 0
    for v in range(0,size[0]-1):
        for u in range(0,size[1]-1):
            kpts[count,0] = v
            kpts[count,1] = u
            count += 1
    return kpts


class KeypointDescTestDataSet(Dataset):
    def __init__(self, test_imdir: str, scale: float = 1.0):

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale        

        test_imnames = list(Path(src_imdir).glob(r'*.png'))
        logging.info(f"{len(test_imnames)} test images are found")
        test_imnames.sort()
        
        self.test_imnames = test_imnames

        if not self.test_imnames:
            raise RuntimeError(
                f'No input file found in {test_imnames}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.test_imnames)} examples')

    def __len__(self):
        return len(self.src_imnames) - 1

    def __getitem__(self, idx):
        '''
        Return a source image, a target image, a set of keypoint location in source image
        '''
        # 设置opencv不使用多进程运行，但这句命令只在本作用域有效。
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        
        src_imname = str(self.src_imnames[idx])
        tgt_imname = str(self.tgt_imnames[idx + 1])

        # src_img = cv2.cvtColor(cv2.imread(src_imname), cv2.COLOR_BGR2RGB)
        src_img = cv2.imread(src_imname)
        # src_img = self.resize_img(src_img)
        # src_img = self.preprocess(src_img, self.scale, is_mask=False,
        #                           is_multi_class=False)
        # tgt_img = cv2.cvtColor(cv2.imread(tgt_imname), cv2.COLOR_BGR2RGB)
        tgt_img = cv2.imread(tgt_imname)
        # tgt_img = self.resize_img(tgt_img)
        # tgt_img = self.preprocess(tgt_img, self.scale, is_mask=False,
        #                           is_multi_class=False)
        
        #'Create keypoints_set by ORB'
        kpts = findTestKeypoints(src_img)

        src_img = src_img.transpose((2, 0, 1)) / 255
        tgt_img = tgt_img.transpose((2, 0, 1)) / 255
        kpts = kpts.transpose((2, 0, 1))

        # print(f"idx:{idx}. src_img:{src_imname}, shape:{src_img.shape}; tgt_img:{tgt_imname}, shape:{tgt_img.shape}")

        return {
            'src_images': torch.as_tensor(src_img.copy()).float().contiguous(),
            'tgt_images': torch.as_tensor(tgt_img.copy()).float().contiguous(),
            'kpts': torch.as_tensor(kpts.copy()).float().contiguous()
        }