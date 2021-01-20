import torch
import torch.nn as nn

class DownSev(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, p=0.1):
        super(DownSev, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=3)
        self.dropout = nn.Dropout(p=p) if p > 0 else None

    def forward(self, x):
        out = self.conv(x)
        if self.dropout:
            out = self.dropout(out)
        return out

    
class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, p=0.1, use_bn=True, use_res=True):
        super(DownSampling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)        
        self.dropout = nn.Dropout(p=p) if p > 0 else None
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):

        out = self.conv1(input)
        if self.dropout:
            out = self.dropout(out)
        out1 = self.bn1(out)
        out = self.relu(out1)

        out = self.conv2(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.bn2(out)
            
        out = out + out1 + input
        out = self.relu(out)
        out = torch.cat((out, input), dim=1)

        return out 


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, cat_channels=None, stride=2, kernel=3, padding=1, p=0.1, use_bn=True, use_res=True):
        super(UpSampling, self).__init__()
        
        if cat_channels == None:
            cat_channels = in_channels+out_channels
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(cat_channels, out_channels, kernel_size=kernel, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=padding)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

        self.bn_deconv = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(p=p) if p > 0 else None
        self.relu = nn.ReLU()

    def forward(self, input, out_down):

        out = self.deconv(input)
        out = self.bn_deconv(out)
        out = self.relu(out)
        out = torch.cat((out_down, out), dim=1)

        out = self.conv1(out)
        if self.dropout:
            out = self.dropout(out)
        out1 = self.bn1(out)
        out = self.relu(out1)

        out = self.conv2(out)
        if self.dropout:
            out = self.dropout(out)
        out2 = self.bn2(out)

        out = self.conv3(out2)
        out = self.bn3(out)

        out = out + out1 + out2
        out = self.relu(out)

        return out


class DRUNet(nn.Module):
    def __init__(self, in_size=3, n_layers=5, root_filters=8, kernal_size=3, pool_size=2, p=0.1, concat_or_add='concat'):
        super(DRUNet, self).__init__()
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
      
        for layer in range(n_layers):
            out_size = 2 ** layer * root_filters
            if layer == 0:
               self.down_layer1 = DownSev(in_size, out_size, p=p)

            self.down_layers.append(DownSampling(out_size, out_size,p=p))

            if layer < n_layers - 1:
                self.pool_layers.append(nn.MaxPool2d(2))

        for layer in range(n_layers - 2, -1, -1):
            out_size = 2 ** (layer + 1) * root_filters
            if layer == 3:
              self.up_layers.append(UpSampling(out_size*2, out_size//2, cat_channels=out_size+out_size//2 ,p=p))
            else: 
              self.up_layers.append(UpSampling(out_size, out_size//2, p=p))

        self.conv_out = nn.Conv2d(root_filters,1, kernel_size=1,stride=1,padding=0)
        self.relu = nn.ReLU()

    def forward(self, input):

        down_tensors = []
        out = self.down_layer1(input)
        out1 = out
        n = len(self.down_layers)

        # down
        for i in range(n):
            down_tensors.append(self.down_layers[i](out))
            out = down_tensors[-1]
            if i < len(self.pool_layers):
                out = self.pool_layers[i](out)
        # up
        for i in range(n-1):  
          out = self.up_layers[i](out, down_tensors[n-2-i])

        out = self.conv_out(out)
        out = self.relu(out)

        return out
