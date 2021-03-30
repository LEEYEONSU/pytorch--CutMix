from torch import nn
import torch.nn.init as init

def conv3x3(in_channels, out_channels, stride = 1):
        return nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(SELayer,self).__init__()
        # AdaptiveAvgPool target size  = 1 x 1 x C 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )
    def forward(self, x):

        out = self.avg_pool(x)
        out = out.view(out.size(0), out.size(1))
        out = self.fc(out)
        out = out.view(out.size(0), out.size(1), 1,1)
        out = out.expand_as(x)
        return x * out

class CifarSEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, reduction = 8):
        super(CifarSEResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SELayer(out_channels, reduction)

        if in_channels != out_channels or stride != 1 :
            self.down_sample = nn.Sequential(
                                                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = stride, bias = False),
                                                nn.BatchNorm2d(out_channels))
        else :  self.down_sample = None

        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.down_sample is not None :
            x = self.down_sample(x)
    
        out = out + x
        out = self.relu(out)

        return out

class CifarSEResNet(nn.Module):        
    def __init__ (self, n_layers, block, num_classes = 10, reduction = 16):
            super(CifarSEResNet, self).__init__()
            self.conv1 = conv3x3(in_channels = 3, out_channels = 16, stride = 1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace = True)
                
            self.layers1 = self._make_layers(block, 16, 16, stride = 1, reduction = reduction)
            self.layers2 = self._make_layers(block, 16, 32, stride = 2, reduction = reduction)
            self.layers3 = self._make_layers(block, 32, 64, stride = 2, reduction = reduction)

            self.avg_pooling = nn.AvgPool2d(8, stride = 1)
            self.fc_out = nn.Linear(64, num_classes)

            for m in self.modules(): 
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)

        # n = # of layers
    def _make_layers(self, block, in_channels, out_channels, stride, reduction, n = 5):

        layers = nn.ModuleList([block(in_channels, out_channels, stride = stride, reduction = reduction)])

        for _ in range(n - 1):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)

        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc_out(out)

        return out

def SEresnet():
    return CifarSEResNet(5, block = CifarSEResidualBlock, reduction = 16)