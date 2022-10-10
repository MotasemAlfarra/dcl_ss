from copy import deepcopy
import torch
from torch import nn
import torch.utils.data
from torch import distributed


class LwF(object):
    def __init__(self, model_old: nn.Module, device='cuda', alpha=0.9, importance=1.0):

        # This implementation is based on:
        #https://github.com/ContinualAI/avalanche/blob/e91f38e3bec5983fc72ac66ee0b39834e13e3e8c/avalanche/training/plugins/lwf.py
        # We do not need a fisher matrix here nor alpha 
        # Alpha here is the temperature parameter for the old prediction

        self.name = "lwf"
        self.device = device
        self.temperature = alpha
        self.model_old = model_old
        self.importance = importance
        # if distributed.get_rank() == 0 :
        #     mymodel = deepcopy(model_old)
        # self.model = mymodel
        # self.model_old.requires_grad_(False)
        # self.model_old.eval()

    def update(self, model, images, labels):
        # self.model_old = model
        return

    def penalty(self, model, images, current_predictions):
        # Distillation loss of the old model at the newly received batch
        old_predictions, _ = self.model_old(images, ret_intermediate=False)
        # print(old_predictions.shape)
        # print(torch.allclose(current_predictions, old_predictions))
        log_p = torch.log_softmax(current_predictions / self.temperature, dim=1)
        q = torch.softmax(old_predictions / self.temperature, dim=1)
        # Changed the batchmean in reduction to mean so that the loss is not too high
        loss = nn.functional.kl_div(log_p, q, reduction="mean")
        return self.importance * loss

    def save_fisher(self, path='./checkpoints/fisher.pth'):
        print("This method does not save fisher. We only need to load the old model")

    def load_fisher(self, path='./checkpoints/fisher.pth'):
        print("This method does not load fisher. We only need to load the old model")