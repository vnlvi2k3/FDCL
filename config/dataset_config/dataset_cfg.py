import numpy as np
import torchio as tio

def dataset_cfg(dataet_name):

    config = {
        'GlaS':
            {
                'PATH_DATASET': 'dataset/GlaS',
                'PATH_TRAINED_MODEL': 'checkpoints',
                'PATH_SEG_RESULT': 'seg_pred',
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.787803, 0.512017, 0.784938],
                'STD': [0.428206, 0.507778, 0.426366],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
            'CRAG':
            {
                'PATH_DATASET': 'dataset/CRAG',
                'PATH_TRAINED_MODEL': 'checkpoints',
                'PATH_SEG_RESULT': 'seg_pred',
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.83065554, 0.72236917, 0.8572969 ],
                'STD': [0.11995327, 0.15234795, 0.09136074],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
        'MoNuSeg':
            {
                'PATH_DATASET': 'dataset/MoNuSeg',
                'PATH_TRAINED_MODEL': 'checkpoints',
                'PATH_SEG_RESULT': 'seg_pred',
                'IN_CHANNELS': 3,
                'NUM_CLASSES': 2,
                'MEAN': [0.64414773, 0.44741015, 0.60391627],
                'STD': [0.18916504, 0.19223037, 0.15346827],
                'PALETTE': list(np.array([
                    [0, 0, 0],
                    [255, 255, 255],
                ]).flatten())
            },
    }

    return config[dataet_name]
