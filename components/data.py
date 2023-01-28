#custom module to create dataset

from torchvision import datasets, transforms
import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset_name, transform=None, train=True, batch_size=64):
    self.cuda = torch.cuda.is_available()
    self.dataset = getattr(datasets, dataset_name)(root='./data',
                                                   download=True, train=train, 
                                                   transform=transform)
    self.dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=64)
    
    self.dataloader = torch.utils.data.DataLoader(self.dataset, **self.dataloader_args)  #returns dataloader when .dataloader is called on Dataset instance.


  def __getitem__(self, index):
        return self.dataset[index]

  def __len__(self):
      return len(self.dataset)
