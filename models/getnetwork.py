import sys
from models import *
import torch.nn as nn


def get_network(network, in_channels, num_classes, **kwargs):
    if network == 'xnet_sb':
        net = XNet_sb(in_channels, num_classes)
    elif network == 'unet':
        net = unet(in_channels, num_classes)
    elif network == 'unet_plusplus' or network == 'unet++':
        net = unet_plusplus(in_channels, num_classes)
    elif network == 'resunet':
        net = res_unet(in_channels, num_classes)
    elif network == 'resunet++':
        net = res_unet_plusplus(in_channels, num_classes)
    elif network == 'swinunet':
        net = swinunet(num_classes, 224)  # img_size = 224
    return net
