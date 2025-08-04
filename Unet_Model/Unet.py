import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = self.conv2D(6,64)
        self.encoder2 = self.conv2D(64,128)
        self.encoder3 = self.conv2D(128,256)
        self.encoder4 = self.conv2D(256,512)
        self.encoder5 = self.conv2D(512,1024)

        self.up_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,kernel_size=2, stride=2)
        self.up_convolution_1 = self.conv2D(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256,kernel_size=2, stride=2)
        self.up_convolution_2 = self.conv2D(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,kernel_size=2,stride=2)
        self.up_convolution_3 = self.conv2D(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=2,stride=2)
        self.up_convolution_4 = self.conv2D(128, 64)

        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        #Encoder
        x1 = self.encoder1(x)
        x2 = self.max_pool(x1)
        x3 = self.encoder2(x2)
        x4 = self.max_pool(x3)
        x5 = self.encoder3(x4)
        x6 = self.max_pool(x5)
        x7 = self.encoder4(x6)
        x8 = self.max_pool(x7)
        x9 = self.encoder5(x8)

        #Decoder
        out = self.up_transpose_1(x9)
        out = self.up_convolution_1(torch.cat((out, x7), dim=1))
        out = self.up_transpose_2(out)
        out = self.up_convolution_2(torch.cat((out, x5), dim=1))
        out = self.up_transpose_3(out)
        out = self.up_convolution_3(torch.cat((out, x3), dim=1))
        out = self.up_transpose_4(out)
        out = self.up_convolution_4(torch.cat((out, x1), dim=1))
        out = self.out(out)

        return out
        

    def conv2D(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        encoder_block = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
                                      nn.ReLU(inplace=True))
        return encoder_block
    
    def loss_fn(self, output, target):
         
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         output = output.to(device)
         optimizer = optim.Adam(self.parameters(), lr=0.001)
         loss = nn.L1Loss()


        