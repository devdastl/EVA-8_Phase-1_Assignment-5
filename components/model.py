# custom module to build network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        if not (kwargs.get("in_channel") or kwargs.get("norm_type") or kwargs.get("drop_out")):
          raise TypeError("please specify in_channel, norm_type and drop_out value")
        
        self.in_channel = kwargs.get("in_channel")
        self.norm = kwargs.get("norm_type")
        self.dropout_value = kwargs.get("drop_out")

      # First block of CNN--------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            self.norm_type(10,26),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # output_size = 26
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), #out_channel reduced from 14 to 12
            self.norm_type(12,24),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # output_size = 24
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), #output_channel reduced from 16 to 14 
            self.norm_type(14,22),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), #in_channel reduced from 16 to 14 
            self.norm_type(10,11),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # output_size = 11

        # Second block of CNN---------------
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), #output_channel reduced from 16 to 14 
            self.norm_type(14,9),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # output_size = 9

        # Third block of CNN---------------
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), #in_channel and output_channel reduced from 16 to 14 
            self.norm_type(14,7),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # output_size = 7

        # Forth block of CNN---------------
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), #in_channel and output_channel reduced from 16 to 14
            self.norm_type(14,5),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        ) # output_size = 5
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False), #in_channel reduced from 16 to 14
        ) # output_size = 5

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) # output_size = 1x1x10 

    #function which setup normalization type based on string input arguments.
    def norm_type(self, feature, map_shape):
      norm_t = self.norm
      group = int(feature/2)
      if (norm_t == "batch"):
        return nn.BatchNorm2d(feature)
      elif (norm_t == "layer"):
        return nn.LayerNorm([feature, map_shape, map_shape], elementwise_affine=False)
      elif (norm_t == "group"):
        return nn.GroupNorm(num_groups=group, num_channels=feature)

      else:
        raise TypeError("please mention normalization technique from batchnorm, layernorm and groupnorm")

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool1(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        x=self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
