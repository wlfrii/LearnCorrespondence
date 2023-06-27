import cv2 as cv
import numpy as np
import math
import utils.util as util
import torch
from utils.cam_param import K, scale


def draw_kpts_all(image, kpts, est_kpts):
    # print(f'kpts.shape:{kpts.shape}')
    # print(f'est_kpts.shape:{est_kpts.shape}')
    image_out = np.concatenate([image, image], axis=1)
    for idx in range(0, kpts.shape[1]-1):
        u = kpts[0,idx,0]
        v = kpts[0,idx,1]

        if abs(u + v) < 1:
            break

        e_u = est_kpts[0,idx,0] + image.shape[1]
        e_v = est_kpts[0,idx,1]

        bgr = np.random.randint(0,255,3,dtype=np.int32)
        # print(bgr)
        color = (np.int(bgr[0]), np.int(bgr[1]), np.int(bgr[2]))

        cv.circle(image_out, (int(u), int(v)), 2, color, -1)
        cv.circle(image_out, (int(e_u), int(e_v)), 2, color, -1)
        cv.line(image_out,(int(u), int(v)),(int(e_u), int(e_v)),color,1,)
    return image_out


def draw_kpts(image, kpts):
    print(f'kpts.shape:{kpts.shape}')
    for idx in range(0, kpts.shape[1]-1):
        u = torch.round(kpts[0,idx,0])
        v = torch.round(kpts[0,idx,1])

        cv.circle(image, (int(u), int(v)), 2, (255, 255, 0), -1)
    return image



def draw_point(image, uv, is_left=True):

    u = uv[0]
    v = uv[1]

    '''Draw the base point'''
    if is_left:
        cv.circle(image, (int(u - 160 * scale), int(v)),
                  3, (255, 255, 255), -1)
    else:
        cv.circle(image, (int(u), int(v)), 3, (255, 255, 255), -1)

    return image

def draw_cor(image, trans, quat, offset=0, K=K):
    if trans[2] <= 0:
        return image

    point = np.mat([
        [trans[0]],
        [trans[1]],
        [trans[2]]
    ])
 
    u, v = space_cor_to_image_cor(point, K)
    
    R = np.array(util.quat2R(quat))
    point_x = point + 10 * np.matmul(R, np.mat([[1], [0], [0]]))
    point_y = point + 10 * np.matmul(R, np.mat([[0], [1], [0]]))
    point_z = point + 10 * np.matmul(R, np.mat([[0], [0], [1]]))

    distance = {
        (255, 0, 0): point_x,
        (0, 255, 0): point_y,
        (0, 0, 255): point_z
    }

    sort = sorted(distance.items(), key=lambda x: x[1][2], reverse=True)

    for axis in sort:
        draw_line(image, point, axis[0], axis[1], offset, K=K)

    '''Draw the base point'''
    cv.circle(image, (int(u - offset * scale), int(v)), 5, (255, 255, 255), -1)
    return image
