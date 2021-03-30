# re-implementation of CutMix
# CutMix, CutMixCrossEntropyLoss

import numpy as np 
from torch.utils.data.dataset import Dataset 

class CutMix(Dataset):
    
    def __init__ (self, dataset, num_class = 10, num_mix = 1, alpha = 1.0, prob = 1.0 ):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.alpha = alpha
        self.prob = prob
    
    def __getitem__(self, idx):
# alphas = torch.from_numpy(np.random.beta(0.4, 0.4

        return data, ohe_label

    def __len__(self):
        return len(self.dataset)