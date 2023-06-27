import torch
from tqdm import tqdm
from corr_loss import correspondence_kpt_loss, correspondence_mask_loss


def evaluate_corr_mask(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    corr_dis_err = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        src_image = batch['src_image']
        tgt_image = batch['tgt_image']
        kpts = batch['kpts']

        # move images and labels to correct device and type
        src_image = src_image.to(device=device, dtype=torch.float32)
        tgt_image = tgt_image.to(device=device, dtype=torch.float32)
        kpts = kpts.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            src_desc = net(src_image)
            tgt_desc = net(tgt_image)
            dis_err,est_kpts = correspondence_mask_loss(src_desc, tgt_desc, kpts)
            corr_dis_err += dis_err
        
    net.train()

    # Fixes a potential division by zero error
    # if num_val_batches == 0:
    #     return loss, trans_loss, rotation_loss
    # return loss / num_val_batches, trans_loss / num_val_batches, rotation_loss / num_val_batches
    if num_val_batches == 0:
        return corr_dis_err
    return corr_dis_err / num_val_batches


def evaluate_corr_kpt(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    corr_dis_err = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        src_image = batch['src_images']
        tgt_image = batch['tgt_images']
        kpts = batch['kpts']

        # move images and labels to correct device and type
        src_image = src_image.to(device=device, dtype=torch.float32)
        tgt_image = tgt_image.to(device=device, dtype=torch.float32)
        kpts = kpts.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the mask
            src_desc = net(src_image)
            tgt_desc = net(tgt_image)
            dis_err,est_kpts = correspondence_kpt_loss(src_desc, tgt_desc, kpts)
            corr_dis_err += dis_err
        
    net.train()

    # Fixes a potential division by zero error
    # if num_val_batches == 0:
    #     return loss, trans_loss, rotation_loss
    # return loss / num_val_batches, trans_loss / num_val_batches, rotation_loss / num_val_batches
    if num_val_batches == 0:
        return corr_dis_err
    return corr_dis_err / num_val_batches