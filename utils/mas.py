from copy import deepcopy
import torch
from torch import nn
import torch.utils.data



class MAS(object):
    def __init__(self, model_old: nn.Module, device='cuda', alpha=0.9, importance=1.0):

        # Fisher matrix is the importance matrix in MAS. 

        self.model_old_dict = deepcopy(model_old.state_dict())
        self.fisher = {}
        self.name = "mas"
        self.importance = importance
        self.device = device
        for n, p in model_old.named_parameters():
            self.fisher[n] = torch.zeros_like(p, device=device, requires_grad=False)

    def update(self, model, images, labels):
        # suppose model have already grad computed, so we can directly update the fisher by getting model.parameters
        self.model_old_dict = deepcopy(model.state_dict())
        model.eval()
        model.zero_grad()
        images, labels = images.to(self.device, dtype=torch.float32), labels.to(self.device, dtype=torch.long)
        output, _ = model(images, ret_intermediate=False)
        loss = torch.norm(output, p=2, dim=1).mean()
        loss.backward()
        for n, p in model.named_parameters():
            print(p.grad.data.abs())
            self.fisher[n] = p.grad.data.clone().abs()
        return

    def penalty(self, model):
        loss = 0.0
        for n, p in model.named_parameters():
            loss += (self.fisher[n].to(p) * (p - self.model_old_dict[n]) ** 2).sum()
        return self.importance * loss

    def save_fisher(self, path='./checkpoints/fisher.pth'):
        torch.save(self.fisher, path)
        print("Fisher matrix is saved in ", path)

    def load_fisher(self, path='./checkpoints/fisher.pth'):
        self.fisher = torch.load(path)
        print("Fisher matrix is loaded from ", path)