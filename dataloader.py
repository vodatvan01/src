import torch.utils.data as data

from util import *
from transforms import *
seed = 2002 

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


import matplotlib.pyplot as plt 
import torch.nn.functional as F

class MattingTransform:
    def __init__(self):
        
        self.matting_transform = {
            "train" : ComposeTrain(),
            "validation" : ComposeValidation()
        }
    def __call__(self, phase, data):
        return self.matting_transform[phase](data)

class MattingDataset(data.Dataset):
    def __init__(self, datasets, phase, transform):
        self.datasets = datasets
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.datasets)
    
    def __getitem__(self, index):
        image ,alpha = self.datasets[index]
        image ,alpha, trimap = self.transform(self.phase, [image ,alpha])
        return image, alpha,trimap 
    
