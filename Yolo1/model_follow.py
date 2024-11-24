import torch
import torch.nn as nn

architecutre_config = [
    
    (7,64,2,3),
    "M",
    (3,192,1,1),
    "M",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "M",
    [(1,256,1,0),(3,512,1,1),4],
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    [(1,512,1,0),(3,1024,1,1),2],
    (3,1024,1,1),
    (3,1024,1,1),
    (3,1024,1,1),
    (3,1024,1,1),
]

class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels, **kwargs):
        super(CNNBlock,self).__init__()
        self.conv =nn.Conv2d(in_channels,out_channels,bias=False,**kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self,x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self,in_channels=3,**kwargs):
        super(Yolov1,self).__init__()
        self.architecture = architecutre_config
        self.in__channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        
    def forward(self,x):
        x = self.darknet(x)
        return self.fc(torch.flatten(x,start_dim =1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in__channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size = x[0], stride = x[2], padding = x[3],
                    )
                ]
                in_channels =x[1]
                
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
                
            elif type(x) ==list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size = conv1[0],
                            stride = conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size =conv2[0],
                            stride = conv2[2],
                            padding= conv2[3],
                        )
                    ]
                    in_channesls = conv2[1]
                    
        return nn.Sequential(*layers)
                    
                    
                    
                


