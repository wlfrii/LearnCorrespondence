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

max_kpt_num = 1000
orb = cv2.ORB_create()


def findKeypoints(image):  
    kpt = orb.detect(image)
    kpts = np.zeros((max_kpt_num, 2, 1), dtype="float")
    for idx in range(0,min(len(kpt)-1, max_kpt_num-1)):
        coord = kpt[idx].pt
        # print(coord)
        kpts[idx,0] = round(coord[0])
        kpts[idx,1] = round(coord[1])
    #print(kpts)
    return kpts


class KeypointDescBinoDataSet(Dataset):
    def __init__(self, src_imdir: str, tgt_imdir: str, scale: float = 1.0):

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale        

        src_imnames = list(Path(src_imdir).glob(r'*.png'))
        tgt_imnames = list(Path(tgt_imdir).glob(r'*.png'))
        assert len(src_imnames) == len(tgt_imnames), \
            "The size of source images should be equal to target images"
        logging.info(f"{len(src_imnames)} source and target images are found")
        src_imnames.sort()
        tgt_imnames.sort()
        self.src_imnames = src_imnames
        self.tgt_imnames = tgt_imnames

        if not self.src_imnames:
            raise RuntimeError(
                f'No input file found in {src_imdir} and {tgt_imdir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.src_imnames)} examples')

    def __len__(self):
        return len(self.src_imnames)

    @staticmethod
    def resize_img(img):
        img = cv2.resize(
            img, (480, 256), interpolation=cv2.INTER_NEAREST)
        return img

    @staticmethod
    def preprocess(img, scale, is_mask, is_multi_class):
        h = img.shape[0]
        w = img.shape[1]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        img = cv2.resize(
            img, (newW, newH), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC)

        if not is_mask:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            img = img / 255
        else:
            if is_multi_class:
                img = img / 10
            else:
                img = img / 255

        return img

    def __getitem__(self, idx):
        '''
        Return a source image, a target image, a set of keypoint location in source image
        '''
        # 设置opencv不使用多进程运行，但这句命令只在本作用域有效。
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        
        src_imname = str(self.src_imnames[idx])
        tgt_imname = str(self.tgt_imnames[idx])

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
        kpts = findKeypoints(src_img)

        src_img = src_img.transpose((2, 0, 1)) / 255
        tgt_img = tgt_img.transpose((2, 0, 1)) / 255
        kpts = kpts.transpose((2, 0, 1))

        # print(f"idx:{idx}. src_img:{src_imname}, shape:{src_img.shape}; tgt_img:{tgt_imname}, shape:{tgt_img.shape}")

        return {
            'src_images': torch.as_tensor(src_img.copy()).float().contiguous(),
            'tgt_images': torch.as_tensor(tgt_img.copy()).float().contiguous(),
            'kpts': torch.as_tensor(kpts.copy()).float().contiguous()
        }


class KeypointDescMonoDataSet(Dataset):
    def __init__(self, src_imdir: str, scale: float = 1.0):

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale        

        src_imnames = list(Path(src_imdir).glob(r'*.png'))
        logging.info(f"{len(src_imnames)} source/target images are found")
        src_imnames.sort()
        self.src_imnames = src_imnames

        if not self.src_imnames:
            raise RuntimeError(
                f'No input file found in {src_imdir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.src_imnames)} examples')

    def __len__(self):
        return len(self.src_imnames)

    @staticmethod
    def resize_img(img):
        img = cv2.resize(
            img, (480, 256), interpolation=cv2.INTER_NEAREST)
        return img

    @staticmethod
    def preprocess(img, scale, is_mask, is_multi_class):
        h = img.shape[0]
        w = img.shape[1]
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        img = cv2.resize(
            img, (newW, newH), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC)

        if not is_mask:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))
            img = img / 255
        else:
            if is_multi_class:
                img = img / 10
            else:
                img = img / 255

        return img

    def __getitem__(self, idx):
        '''
        Return a source image, a target image, a set of keypoint location in source image
        '''
        # 设置opencv不使用多进程运行，但这句命令只在本作用域有效。
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        
        src_imname = str(self.src_imnames[idx])
        ridx = np.random.randint(-30,30,1,dtype=np.int32)
        tgt_idx = min(max(0, idx + np.int(ridx)), len(self.src_imnames)-1)
        # print(f'ridx:{ridx}, tgt_idx:{tgt_idx}')
        tgt_imname = str(self.src_imnames[tgt_idx])

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
        kpts = findKeypoints(src_img)

        src_img = src_img.transpose((2, 0, 1)) / 255
        tgt_img = tgt_img.transpose((2, 0, 1)) / 255
        kpts = kpts.transpose((2, 0, 1))

        # print(f"idx:{idx}. src_img:{src_imname}, shape:{src_img.shape}; tgt_img:{tgt_imname}, shape:{tgt_img.shape}")

        return {
            'src_images': torch.as_tensor(src_img.copy()).float().contiguous(),
            'tgt_images': torch.as_tensor(tgt_img.copy()).float().contiguous(),
            'kpts': torch.as_tensor(kpts.copy()).float().contiguous()
        }