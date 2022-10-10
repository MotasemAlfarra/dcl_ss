from copy import deepcopy
import torch
from torch import nn
import torch.utils.data



class EWC(object):
    def __init__(self, model_old: nn.Module, device='cuda', alpha=0.9, importance=1.0):

        self.model_old_dict = deepcopy(model_old.state_dict())
        self.alpha = alpha
        self.fisher = {}
        self.name = "ewc"
        self.importance = importance
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
        self.device = device
        for n, p in model_old.named_parameters():  # update fisher with new keys (due to incremental classes)
            self.fisher[n] = torch.ones_like(p, device=device, requires_grad=False)

    def update(self, model, images, labels):
        model.eval()
        model.zero_grad()
        images, labels = images.to(self.device, dtype=torch.float32), labels.to(self.device, dtype=torch.long)
        output, _ = model(images, ret_intermediate=False)
        loss = self.criterion(output, labels).mean()
        loss.backward()
        # suppose model have already grad computed, so we can directly update the fisher by getting model.parameters
        for n, p in model.named_parameters():
            # print(p.grad)
            self.fisher[n] = (self.alpha * (p.grad ** 2)) + ((1 - self.alpha) * self.fisher[n].to(p))
        model.zero_grad()
        
    def penalty(self, model, images=None, current_predictions=None):
        loss = 0.
        for n, p in model.named_parameters():
            loss += (self.fisher[n].to(p) * (p - self.model_old_dict[n]) ** 2).sum()
        return self.importance * loss

    def save_fisher(self, path='./checkpoints/fisher.pth'):
        torch.save(self.fisher, path)
        print("Fisher matrix is saved in ", path)

    def load_fisher(self, path='./checkpoints/fisher.pth'):
        self.fisher = torch.load(path)
        print("Fisher matrix is loaded from ", path)