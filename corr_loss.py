import torch
import cv2
import torch.nn as nn
import numpy as np
from utils.util import tensor2opencv

# def findMax(mat):
#     assert mat.ndim == 2, 'The input mat should be 2D'
#     height = mat.shape[0]
#     width = mat.shape[1]
#     idx = torch.argmax(mat)
#     if ((idx + 1) % width) == 0:
#         v = (idx + 1) // width - 1
#         u = width - 1
#     else:
#         v = (idx + 1) // width
#         u = (idx + 1) % width - 1
#     return u,v

def findMax(mat):
    assert mat.ndim == 2, 'The input mat should be 2D'  
    idx = torch.nonzero(torch.ge(mat,torch.max(mat)))
    u = idx[0][1]
    v = idx[0][0]
    return u,v


def correspondence_mask_loss(src_desc, tgt_desc, kpts):
    # print(f'kpts.shape:{kpts.shape}')
    # print(f'src_desc.shape:{src_desc.shape}')
    # print(f'tgt_desc.shape:{tgt_desc.shape}')
    corr_loss = 0  
    est_kpts = kpts
    count = 0
    for c in range(0, kpts.shape[0]-1):
        for v in range(0, kpts.shape[2]-1):
            for u in range(0, kpts.shape[3]-1):
            
                if kpts[c,0,v,u] < 0.5:
                    continue

                count += 1

                #'Find desc in source'
                uv_desc = src_desc[c,:,v,u]
                # print(f'uv_desc.shape:{uv_desc.shape}')
                desc_mat = uv_desc.repeat(tgt_desc.shape[2],
                                          tgt_desc.shape[3], 1)
                desc_mat = desc_mat.permute(2,0,1)
                # print(f'desc_mat.shape:{desc_mat.shape}')
                #'Calc target heatmap'
                mat_dot = torch.mul(desc_mat, tgt_desc[c,:,:,:])
                # print(f'mat_dot.shape:{mat_dot.shape}')
                mat = torch.sum(mat_dot, dim=0)
                # print(f'mat.shape:{mat.shape}')

                #'Find desc in target'
                ut,vt = findMax(mat)
                uv_desc = tgt_desc[c,:,vt,ut]
                desc_mat = uv_desc.repeat(tgt_desc.shape[2],
                                          tgt_desc.shape[3], 1)
                desc_mat = desc_mat.permute(2,0,1)
                #'Calc source heatmap'
                mat_dot = torch.mul(desc_mat, src_desc[c,:,:,:])
                mat = torch.sum(mat_dot, dim=0)
                #'Calc loss'
                us,vs = findMax(mat)
                loss = torch.norm(torch.tensor([u,v]).float() - torch.tensor([us,vs]).float())
                # print(f'u,v:[{u},{v}], us,vs:[{us},{vs}], loss:{loss}')
                corr_loss += loss

                est_kpts[c,0,v,u] = us
                est_kpts[c,0,v,u] = vs

    return corr_loss / count

sigma = 10

def correspondence_kpt_loss(src_desc, tgt_desc, kpts):
    # print(f'kpts.shape:{kpts.shape}')
    # print(f'src_desc.shape:{src_desc.shape}')
    # print(f'tgt_desc.shape:{tgt_desc.shape}')
    corr_loss = 0  
    est_kpts = kpts
    count = 0
    for c in range(0, kpts.shape[0]-1):
        for idx in range(0, kpts.shape[2]-1):
            u = torch.round(kpts[c,0,idx,0])
            v = torch.round(kpts[c,0,idx,1])
            if abs(u + v) < 1:
                break

            count += 1

            #'Find desc in source'
            uv_desc = src_desc[c,:,v,u]
            # print(f'uv_desc.shape:{uv_desc.shape}')
            desc_mat = uv_desc.repeat(tgt_desc.shape[2],
                                        tgt_desc.shape[3], 1)
            desc_mat = desc_mat.permute(2,0,1)

            # print(f'v,u:[{v},{u}]')
            # print(f'uv_desc:{uv_desc}')
            # print(f'desc_mat(:,v,u):{desc_mat[:,v,u]}')
            # print(f'desc_mat(:,100,100):{desc_mat[:,100,100]}')

            # print(f'desc_mat.shape:{desc_mat.shape}')
            #'Calc target heatmap'
            mat_dot = torch.mul(desc_mat, tgt_desc[c,:,:,:])
            # print(f'mat_dot.shape:{mat_dot.shape}')
            mat = torch.sum(mat_dot, dim=0)
            img = mat.cpu().detach().numpy()*255
            img = img.astype(np.uint8)
            if v == 135 & u == 240:
                cv2.imwrite(f'./u_{u}_v_{v}_heatmap_target.png', img)
            # print(f'mat.shape:{mat.shape}')

            # print(f'mat_dot(:,v,u):{mat_dot[:,v,u]}')
            # print(f'mat(v,u):{mat[v,u]}')
            # print(f'uv_desc.dot: {torch.dot(uv_desc, uv_desc)}')

            #'Find desc in target'
            ut,vt = findMax(mat)
            uv_desc = tgt_desc[c,:,vt,ut]
            desc_mat = uv_desc.repeat(tgt_desc.shape[2],
                                        tgt_desc.shape[3], 1)
            desc_mat = desc_mat.permute(2,0,1)
            #'Calc source heatmap'
            mat_dot = torch.mul(desc_mat, src_desc[c,:,:,:])
            mat = torch.sum(mat_dot, dim=0)
            img = mat.cpu().detach().numpy()*255
            img = img.astype(np.uint8)
            if v == 135 & u == 240:
                cv2.imwrite(f'./u_{u}_v_{v}_heatmap_est_source.png', img)
            #'Calc loss'
            us,vs = findMax(mat)
            loss = torch.norm(torch.tensor([u,v]).float() - torch.tensor([us,vs]).float())
            loss2 = torch.norm(torch.tensor([u,v]).float() - torch.tensor([ut,vt]).float())
            corr_loss += loss + 0.4*loss2

            # '''Relative response loss'''
            # # The heatmap value in (u,v) location
            # loss_uv = torch.exp(mat[v,u] * sigma)
            # loss_rr_sum = 0
            # # The possible heatmap values for (u,v) location
            # rr_idx = torch.nonzero(torch.ge(mat, 0.9*mat[vs,us]))
            # for rr_id in rr_idx:
            #     M_rr = mat[rr_id[0], rr_id[1]]
            #     loss_rr_sum += torch.exp(M_rr * sigma)
            # # Relative response loss
            # loss_rr = -100*torch.log(loss_uv / loss_rr_sum) 
            # corr_loss += loss_rr + loss

            # print(f'{count} u,v:[{u},{v}], us,vs:[{us},{vs}], loss:{loss}, loss_rr:{loss_rr}')

            est_kpts[c,0,idx,0] = us
            est_kpts[c,0,idx,1] = vs
    # print(f'kpt num: {count}')
    return corr_loss / count, est_kpts
    


'input is src_desc, tgt_desc, and kpt_mask'
class Corr_Loss(nn.Module):
    def __init__(self):
        super(Corr_Loss, self).__init__()
    
    def forward(self, src_desc, tgt_desc, kpt_mask):
        loss = correspondence_loss(src_desc, tgt_desc, kpts)
        return loss