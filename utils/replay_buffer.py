from numpy import float32
import torch

class Replay_Buffer(object):
    """"
    This class implements First In First Out (FIFO) Buffer 
    for memory based continual learning techniques
    """
    def __init__(self, buffer_size, batch_size) -> None:
        
        self.current_replace = 0
        self.current_buffer_size = 0
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        # I am storing the images in the RAM and will move only batches to GPU when training
        self.image_buffer = torch.zeros((buffer_size, 3, 512, 512), dtype=torch.float32, device='cpu')
        self.label_buffer = torch.zeros((buffer_size, 512, 512), dtype=torch.long, device='cpu')

    def update(self, images, labels):
        batch_size = images.size(0)
        # Update the buffer size
        self.current_buffer_size += batch_size
        self.current_buffer_size = min(self.current_buffer_size, self.buffer_size)
        # Update the buffer images and labels
        self.image_buffer[self.current_replace : self.current_replace + batch_size] = images
        self.label_buffer[self.current_replace : self.current_replace + batch_size] = labels
        # Update the index in which we are updating the buffer (FIFO)
        self.current_replace += batch_size
        if self.current_replace > self.buffer_size-1:
            self.current_replace = 0
        return

    def sample(self):
        #Sample non-pverlapping set of indices
        indxs = torch.randperm(self.current_buffer_size)[:self.batch_size]
        return self.image_buffer[indxs], self.label_buffer[indxs]