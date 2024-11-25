import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchio import transforms as T
import torchio as tio

def data_transform_2d():
    data_transforms = {
        'train': A.Compose([
            A.Resize(224, 224, p=1),
            A.Flip(p=0.75),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=1),
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
        ),
        'val': A.Compose([
            A.Resize(224, 224, p=1),
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
        ),
        'test': A.Compose([
            A.Resize(224, 224, p=1),
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
        )
    }
    return data_transforms

def data_normalize_2d(mean, std):
    data_normalize = A.Compose([
            A.Normalize(mean, std),
            ToTensorV2()
        ],
            additional_targets={'image2': 'image', 'mask2': 'mask'}
    )
    return data_normalize
