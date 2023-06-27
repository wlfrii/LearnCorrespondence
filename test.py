import argparse
import os
import logging
from pathlib import Path
import torch
import cv2
import time
import numpy as np
import math
import pickle
from models.unet_model import UNet
# from corr_dataset_tests import KeypointDescTestDataSet


def draw_predict_kpts(src_image, tgt_image, correspondence):
    # print(f'kpts.shape:{kpts.shape}')
    # print(f'est_kpts.shape:{est_kpts.shape}')
    image_out = np.concatenate([src_image, tgt_image], axis=1)
    for idx in range(0, len(correspondence)-1):
        vs = correspondence[idx][0]
        us = correspondence[idx][1]
        vt = correspondence[idx][2]
        ut = correspondence[idx][3] + src_image.shape[1]

        bgr = np.random.randint(0,255,3,dtype=np.int32)
        # print(bgr)
        color = (np.int(bgr[0]), np.int(bgr[1]), np.int(bgr[2]))

        cv2.circle(image_out, (int(us), int(vs)), 2, color, -1)
        cv2.circle(image_out, (int(ut), int(vt)), 2, color, -1)
        cv2.line(image_out,(int(us), int(vs)),(int(ut), int(vt)),color,1,)
    return image_out

def findMax(mat):
    assert mat.ndim == 2, 'The input mat should be 2D'  
    idx = torch.nonzero(torch.ge(mat,torch.max(mat)))
    u = idx[0][1]
    v = idx[0][0]
    return u,v

def to_tensor(image):
    image = torch.as_tensor(image.copy()).float().contiguous()
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)
    return image

def corr_condition(src_pt, tgt_pt):
    if abs(src_pt[0] - tgt_pt[0]) < 15:
        if abs(src_pt[1] - tgt_pt[1]) < 15:
            return True
    return False

def find_corr_index(src_desc, tgt_desc, interval = 10):
    correspondence = []
    for v in range(0, src_desc.shape[2]-1, interval):
        for u in range(0, src_desc.shape[3]-1, interval):
            uv_desc = src_desc[0,:,v,u]
            desc_mat = uv_desc.repeat(tgt_desc.shape[2],
                                      tgt_desc.shape[3], 1)
            desc_mat = desc_mat.permute(2,0,1)
            mat_dot = torch.mul(desc_mat, tgt_desc[0,:,:,:])
            mat = torch.sum(mat_dot, dim=0)
            ut,vt = findMax(mat)

            uv_desc = tgt_desc[0,:,vt,ut]
            desc_mat = uv_desc.repeat(tgt_desc.shape[2],
                                        tgt_desc.shape[3], 1)
            desc_mat = desc_mat.permute(2,0,1)
            #'Calc source heatmap'
            mat_dot = torch.mul(desc_mat, src_desc[0,:,:,:])
            mat = torch.sum(mat_dot, dim=0)
            us,vs = findMax(mat)
            loss = torch.norm(torch.tensor([u,v]).float() - torch.tensor([us,vs]).float())

            if loss < 2:
                if corr_condition([v,u], [vt,ut]):
                    correspondence += [[v,u,vt,ut]]
    print(f"Learned correspondence.len: {len(correspondence)}")
    return correspondence

def predict_correspondence(net, src_image, tgt_image, device, 
                           write_features = False, filename = None):
    net.eval()

    src_image_t = to_tensor(src_image.transpose((2, 0, 1)) / 255)
    tgt_image_t = to_tensor(tgt_image.transpose((2, 0, 1)) / 255)

    if write_features:
        interval = 2
    else:
        interval = 4

    with torch.no_grad():
        src_desc = net(src_image_t)
        tgt_desc = net(tgt_image_t)
        # print(f'src_desc.shape:{src_desc.shape}')
        # print(f'tgt_desc.shape:{tgt_desc.shape}')

        correspondence = find_corr_index(src_desc, tgt_desc, interval = interval)
        pred_image = draw_predict_kpts(src_image, tgt_image, correspondence)

        if write_features & (filename != None):
            print(f"Write features to file: {filename}.txt")
            with open(filename+".txt", 'w', encoding='utf-8') as fid:
                for line in correspondence:
                    fid.write(str(line) + '\n')

    return pred_image, len(correspondence)

orb = cv2.ORB_create()
def predict_orb_correspondence(src_image, tgt_image,
                               write_features = False, filename = None):
    kpt1, desc1 = orb.detectAndCompute(src_image, None)
    kpt2, desc2 = orb.detectAndCompute(tgt_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    correspondence = []
    for m in matches:
        src_pt = kpt1[m.queryIdx].pt
        tgt_pt = kpt2[m.trainIdx].pt

        if corr_condition(src_pt, tgt_pt):
            correspondence += [[src_pt[0], src_pt[1], tgt_pt[0], tgt_pt[1]]]
    print(f"ORB correspondence.len: {len(correspondence)}")
    pred_image = draw_predict_kpts(src_image, tgt_image, correspondence)

    if write_features & (filename != None):
        print(f"Write ORB features to file: {filename}_orb.txt")
        with open(filename+"_orb.txt", 'w', encoding='utf-8') as fid:
            for line in correspondence:
                fid.write(str(line) + '\n')

    return pred_image, len(correspondence)


desc_len = 128
# trained_model_path = "./pth/Correspondence/KeypointDescNet_20230613_1554.pth"
# test_image_folder = "../data/test/"
trained_model_path = "./pth/Correspondence/KeypointDescNet_20230614.pth"
# test_image_folder = "../data/left/"
# test_image_range = range(298, min(len(test_imnames)-1, 2000000))
test_image_folder = "../data/test/"
test_image_range = range(1, 20)
is_write_features = True

if __name__ == '__main__':

    kpt_desc_net = UNet(n_channels=3,n_classes=desc_len)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}') 

    kpt_desc_net.to(device=device)
    kpt_desc_net.load_state_dict(torch.load(trained_model_path, map_location=device))
    print(f'KeypointDesc model loaded from {trained_model_path}')

    test_imnames = os.listdir(test_image_folder)
    test_imnames.sort()
  
    for idx in test_image_range:
        
        print(f"Process {test_imnames[idx]}")
        src_image = cv2.imread(test_image_folder + str(test_imnames[idx]))
        tgt_image = cv2.imread(test_image_folder + str(test_imnames[idx+1]))

        name_idx = str(test_imnames[idx][:-4]) + "__" + str(test_imnames[idx+1][:-4])
        corr_data_filename = test_image_folder[:-1] + "_correspondence/"+name_idx

        pred_image, len1 = predict_correspondence(kpt_desc_net, src_image=src_image, tgt_image=tgt_image, device=device, write_features=is_write_features, filename=corr_data_filename)
        pred_image2, len2 = predict_orb_correspondence(src_image=src_image, tgt_image=tgt_image, write_features=is_write_features, filename=corr_data_filename)

        pred_image = np.concatenate([pred_image, pred_image2], axis=0)
        cv2.imwrite('./test/in-vivo1/'+name_idx+'.png', pred_image)

    print("Test done")