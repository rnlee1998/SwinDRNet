
from genericpath import isdir
import torch
import cv2
import os
import numpy as np
from networks.SwinDRNet import SwinDRNet
from trainer import SwinDRNetTrainer
from config import get_config
import matplotlib.pyplot as plt
from tqdm import tqdm
import imgaug as ia
import argparse
import logging
import random
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from imgaug import augmenters as iaa
from torchvision import transforms
import torch.nn.functional as F
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

augs_test = iaa.Sequential([
    iaa.Resize({
        "height": 224,
        "width": 224
    }, interpolation='nearest'),
])

def transfer_to_device(_trainer, sample_batched):
        for key in sample_batched:
            sample_batched[key] = sample_batched[key].to(_trainer.device)
        return sample_batched

def SingleInference(_trainer,rgb_path,depth_path,sample_mode='bilinear'):
    print('\nDemo single Test:')
    print('=' * 10)
    _trainer.model.eval()
    custom_sample_batched={}
    if os.path.isfile(rgb_path):
        if rgb_path[-3:]=='jpg' or rgb_path[-3:]=='png':
            _rgb = Image.open(rgb_path).convert('RGB')
            _rgb = np.array(_rgb)
        else:
            _rgb = np.load(rgb_path)
    else:
        _rgb = rgb_path

    h,w,_ = _rgb.shape
    _rgb = augs_test.to_deterministic().augment_image(_rgb)
    _rgb_tensor = transforms.ToTensor()(_rgb)
    _rgb_tensor = _rgb_tensor.unsqueeze(0)
    custom_sample_batched['rgb'] = _rgb_tensor

    if os.path.isfile(depth_path):
        if depth_path[-3:]=='npz' or depth_path[-3:]=='npy':#单位是mm
            _sim_depth =  np.load(depth_path) /1000.0   #把单位化为m
        else:
            _sim_depth =  cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)#单位是m
    else:
        _sim_depth = depth_path
    # print(_sim_depth.mean(),_sim_depth.var())
    if len(_sim_depth.shape) == 3:
        _sim_depth = _sim_depth[:, :, 0]
    _sim_depth = _sim_depth[np.newaxis, ...]
    _sim_depth = _sim_depth.transpose((1, 2, 0))  # To Shape: (H, W, 1)
    # transform to xyz_img
    _sim_depth = augs_test.to_deterministic().augment_image(_sim_depth, hooks=ia.HooksImages())  
    _sim_depth = _sim_depth.transpose((2, 0, 1))  # To Shape: (1, H, W)
    _sim_depth[_sim_depth <= 0] = 0.0
    _sim_depth = _sim_depth.squeeze(0)            # (H, W)
    _sim_depth_tensor = transforms.ToTensor()(np.uint8(_sim_depth))
    _sim_depth_tensor = _sim_depth_tensor.unsqueeze(0) 
    custom_sample_batched['sim_depth'] = _sim_depth_tensor

    custom_sample_batched = transfer_to_device(_trainer,custom_sample_batched)

    with torch.no_grad():  
        outputs_depth = _trainer.forward(custom_sample_batched, mode='single_test')
        if  outputs_depth.shape[2:] != (h,w):
            #upsampling to origin resolution#
            outputs_depth = F.interpolate(outputs_depth,(h,w),mode=sample_mode)
  
        outputs_depth = np.array(1000*outputs_depth.cpu()).astype(np.uint16)
        return outputs_depth


def DemoInference(_trainer,data_path):
    print('\nDemo batch Test:')
    print('=' * 10)
    _trainer.model.eval()
    filelist = os.listdir(data_path)
    custom_sample_batched={}
    for item in tqdm(filelist):
        item_path = os.path.join(data_path,item)
        if '_rgb.png' in item:   
            rgb_path = item_path 
            depth_path = rgb_path.replace('_rgb.png','_depth.png')     

            _rgb = Image.open(item_path).convert('RGB')
            _rgb = np.array(_rgb)
            _rgb = augs_test.to_deterministic().augment_image(_rgb)
            _rgb_tensor = transforms.ToTensor()(_rgb)
            _rgb_tensor = _rgb_tensor.unsqueeze(0)
            custom_sample_batched['rgb'] = _rgb_tensor

            _sim_depth =  cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if len(_sim_depth.shape) == 3:
                _sim_depth = _sim_depth[:, :, 0]
            _sim_depth = _sim_depth[np.newaxis, ...]
            _sim_depth = _sim_depth.transpose((1, 2, 0))  # To Shape: (H, W, 1)
            # transform to xyz_img
            _sim_depth = augs_test.to_deterministic().augment_image(_sim_depth, hooks=ia.HooksImages())  
            _sim_depth = _sim_depth.transpose((2, 0, 1))  # To Shape: (1, H, W)
            _sim_depth[_sim_depth <= 0] = 0.0
            _sim_depth = _sim_depth.squeeze(0)            # (H, W)
            _sim_depth_tensor = transforms.ToTensor()(np.uint8(_sim_depth))
            _sim_depth_tensor = _sim_depth_tensor.unsqueeze(0) 
            custom_sample_batched['sim_depth'] = _sim_depth_tensor

            custom_sample_batched = transfer_to_device(_trainer,custom_sample_batched)

            with torch.no_grad():  
                outputs_depth = _trainer.forward(custom_sample_batched, mode='single_test')
                outputs_depth = np.array(1000*outputs_depth.cpu()).astype(np.uint16)
                # plt.imsave(os.path.join('/data/ran1998.li/workdir/DREDS-main/SwinDRNet/results/DREDS_CatKnown/depth/','%s.png'%(i)), outputs_depth, cmap='PuBu')
                cv2.imwrite(os.path.join('/data/ran1998.li/workdir/DREDS_main/SwinDRNet/results/Demo/save/',
                                            '%s_predict.png'%(item[:-4])), outputs_depth[0,0,:,:])


parser = argparse.ArgumentParser()
parser.add_argument('--mask_transparent', action='store_true', default=True, help='material mask')
parser.add_argument('--mask_specular', action='store_true', default=True, help='material mask')
parser.add_argument('--mask_diffuse', action='store_true', default=True, help='material mask')

parser.add_argument('--train_data_path', type=str,
                    default='/data/ran1998.li/datasets/DREDS/DREDS-CatKnown/train/', help='root dir for training dataset')

parser.add_argument('--val_data_path', type=str,
                    default='/data/ran1998.li/datasets/DREDS/DREDS-CatKnown/val/', help='root dir for validation dataset')
parser.add_argument('--val_data_type', type=str,
                    default='sim', help='type of val dataset (real/sim)')

parser.add_argument('--output_dir', type=str, 
                    default='/data/ran1998.li/workdir/DREDS_main/SwinDRNet/results/DREDS_CatKnown', help='output dir')
parser.add_argument('--checkpoint_save_path', type=str, 
                    default='models/DREDS', help='Choose a path to save checkpoints')

parser.add_argument('--decode_mode', type=str, 
                    default='multi_head', help='Select encode mode')
parser.add_argument('--val_interation_interval', type=int, 
                    default=5000, help='The iteration interval to perform validation')

parser.add_argument('--percentageDataForTraining', type=float, 
                    default=1.0, help='The percentage of full training data for training')
parser.add_argument('--percentageDataForVal', type=float, 
                    default=1.0, help='The percentage of full training data for training')

parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')

parser.add_argument('--cfg', type=str, 
            default="/data/ran1998.li/workdir/DREDS_main/SwinDRNet/configs/swin_tiny_patch4_window7_224_lite.yaml", 
            metavar="FILE", 
            help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', default=True, help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume',type=str, default='./output-1/epoch_149.pth', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')


args = parser.parse_args()
config = get_config(args)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device_list = [0]
model_path = "models/model.pth"


def batchtest(_trainer):
    data_path = "/data/ran1998.li/workdir/DREDS_main/SwinDRNet/results/Demo/depth/"
    DemoInference(_trainer,data_path)

def singletest(_trainer,rgb_path,depth_path,sample_mode='bilinear'):
    return SingleInference(_trainer,rgb_path,depth_path,sample_mode)

def init_trainer(model_path):
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    net = SwinDRNet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    trainer = SwinDRNetTrainer
    _trainer = trainer(args, net, device_list, model_path)
    return _trainer

if __name__ == "__main__":
    _trainer = init_trainer()
    batchtest(_trainer)