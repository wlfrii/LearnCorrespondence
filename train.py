import argparse
import logging
from pathlib import Path

import torch
import wandb
from torch import optim
from tqdm import tqdm
import cv2

# Local
from models.KeypointDescNet import KeypointDescriptorNet
from models.unet_model import UNet
from models.ResNet import resnet34, resnet50, resnet101
from utils.util import tensor2opencv

from corr_dataset import KeypointDescBinoDataSet, KeypointDescMonoDataSet
from corr_loss import correspondence_kpt_loss, correspondence_mask_loss
from corr_evaluate import evaluate_corr_kpt, evaluate_corr_mask
from corr_draw import draw_kpts,draw_kpts_all

is_use_mono_dataset = True
src_imdir = '../data/left/'
tgt_imdir = '../data/right/'
mono_imdir = '../data/train_sequence_01/'
desc_len = 128

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 2,
              learning_rate: float = 1e-5,
              valid_percent: float = 0.2,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              is_syn: bool = True,
              bg_num: int = 30):
    project = "Correspondence"

    # 1. Create dataset
    if is_use_mono_dataset:
        dataset = KeypointDescMonoDataSet(src_imdir=mono_imdir)
        logging.info('Using mono dataset')
    else:
        dataset = KeypointDescBinoDataSet(src_imdir=src_imdir, tgt_imdir=tgt_imdir)
        logging.info('Using bino dataset')

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * valid_percent)
    n_train = len(dataset) - n_val
    train_set, valid_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    logging.info(f"train_set.size:{len(train_set)}, valid_set.size:{len(valid_set)}")

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    valid_loader = torch.utils.data.DataLoader(valid_set, shuffle=False,
                            drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project=project, resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  valid_percent=valid_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    
    # Change learning-rate
    # 'min'模型检测metric是否不再减小，'max'模型检测metric是否不再增大
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', factor=0.5, patience=2, min_lr=1e-10, eps=1e-10)
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    
    dir_checkpoint = Path('./checkpoints/PositionNet')
    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                src_images = batch['src_images']
                tgt_images = batch['tgt_images']
                kpts = batch['kpts']

                # print(src_images.shape)
                # print(tgt_images.shape)

                assert src_images.shape[1] == 3 & tgt_images.shape[1] == 3, \
                    f'Network has been defined with 3 input channels, ' \
                    f'but loaded images have {src_images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                src_images = src_images.to(device=device, dtype=torch.float32)
                tgt_images = tgt_images.to(device=device, dtype=torch.float32)
                kpts = kpts.to(device=device, dtype=torch.long)
                
                with torch.cuda.amp.autocast(enabled=amp):
                    src_desc = net(src_images)
                    tgt_desc = net(tgt_images)
                    loss,est_kpts = correspondence_kpt_loss(src_desc, tgt_desc, kpts)
                    
                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10, norm_type=2)
                optimizer.step()

                pbar.update(src_images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:

                        curr_kpts = kpts[0].cpu().detach().numpy()
                        curr_est_kpts = est_kpts[0].cpu().detach().numpy()

                        coor_dis_err = evaluate_corr_kpt(net, valid_loader, device)
                        scheduler.step(coor_dis_err)

                        logging.info(
                            'Average corr distance err (pixel): {}'.format(coor_dis_err))
                        
                        # src_img_with_kpts = draw_kpts(tensor2opencv(src_images[0]), curr_kpts)
                        # est_src_img_with_kpts = draw_kpts(tensor2opencv(src_images[0]), curr_est_kpts)
                        est_src_img_with_kpts = draw_kpts_all(tensor2opencv(src_images[0]), curr_kpts, curr_est_kpts)

                        cv2.imwrite('./valid/'+str(desc_len)+'/Epoch'+str(epoch)+'Step'+str(global_step)+'.png', est_src_img_with_kpts)
                        
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'avg corr distance err': coor_dis_err,
                            # 'src_img_with_kpts': src_img_with_kpts,
                            'est_src_img_with_kpts': wandb.Image(est_src_img_with_kpts),
                            'step': global_step,
                            'epoch': epoch,
                        })
                        dir_checkpoint = Path('./checkpoints/Correspondence')

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint /
                       'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the net on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E',
                        type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size',
                        metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float,
                        default=0.25, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.2,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--classes', '-c', type=int,
                        default=2, help='Number of classes')
    parser.add_argument('--syn', action='store_false',
                        default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #original_model = resnet34(pretrained=True, in_channel=3)
    # print(original_model)
    #net = KeypointDescriptorNet(original_model, mid_channels=1024)
    net = UNet(n_channels=3,n_classes=desc_len)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  valid_percent=args.val,
                  is_syn=args.syn)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
