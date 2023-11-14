import torch
import torch.nn as nn
import basicblock as B
import torch.nn.functional as F
import numpy as np
'''
# ====================
# T-net
# ====================
'''


# ====================
# Position attention block
# ====================
class PAB(nn.Module):
    def __init__(self, in_channels):
        super(PAB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels//8, 1, 1, 0, bias=True)
        self.act = nn.ReLU(inplace=False)
    def forward(self, x, C, H ,W):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x3 = x1.reshape(C,H*W)
        x4 = x2.reshape(C//8,H*W)
        x5 = x4.transpose(0,1)
        x6 = F.softmax(torch.matmul(x5,x4)).clone()
        x7 = torch.matmul(x3, x6)
        x8 = x7.reshape(C,W,H)
        return x8 + x

# ====================
# Channel attention block
# ====================
class CAB(nn.Module):
    def __init__(self, nc=4, gc=16, kernel_size=3, stride=1, padding=1, bias=True, mode='CR', negative_slope=0.2):
        super(CAB, self).__init__()

    def forward(self, x, C, H, W):
        x1 = x.reshape(C,H*W)
        x2 = x1.transpose(0,1)
        x3 = F.softmax(torch.matmul(x1, x2)).clone()
        x4 = x.reshape(C,H*W)
        x5 = torch.matmul(x3,x4)
        x6 = x5.reshape(C, W, H)
        return x6 + x

# ====================
# Dual attention Module
# ====================
class DAM(nn.Module):
    def __init__(self):
        super(DAM, self).__init__()
        self.conv1 = nn.Conv2d(1024, 1024, 1, 1, 0, bias=True)
        self.pab = PAB(in_channels=1024)
        self.cab = CAB()
        self.act = nn.ReLU(inplace=False)
    def forward(self, x):
        x1 = self.act(self.conv1(x))
        C = x1.size()[1]
        W = x1.size()[2]
        H = x1.size()[3]
        x2 = self.pab(x1, C, H, W)
        x3 = self.cab(x1, C, H, W)
        x4 = x2 + x3
        return x2 + x3




# ====================
# Residual dense block
# ====================
class RDB(nn.Module):
    def __init__(self, nc=4, gc=16, kernel_size=3, stride=1, padding=1, bias=True, mode='Cr', negative_slope=0.2):
        super(RDB, self).__init__()
        # gc: growth channel
        self.conv1 = B.conv(nc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv2 = B.conv(nc+gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv3 = B.conv(nc+2*gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
        self.conv4 = B.conv(nc+3*gc, nc, 1, stride, 0, bias, mode, negative_slope)

#        self.conv4 = B.conv(nc+3*gc, gc, kernel_size, stride, padding, bias, mode, negative_slope)
 #       self.conv5 = B.conv(nc+4*gc, nc, 1, stride, 0, bias, mode, negative_slope)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
#        x5 = self.conv5(torch.cat((x, x1, x2, x3,x4), 1))
        return x4 + x


# ====================
# Downsampling block
# ====================
class Downsampling(nn.Module):
    def __init__(self, in_channels):
        super(Downsampling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=True)   # in_channels, out_channels, kernel_size, stride, padding
        self.conv2 = nn.Conv2d(in_channels, in_channels*2 , 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=False)
    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        return x2


# ====================
# Upsampling block
# ====================
class Upsampling(nn.Module):
    def __init__(self, in_channels):
        super(Upsampling, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0, bias=True)
        self.conv1 = nn.Conv2d(in_channels, int(in_channels/2) , 3, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x1 = self.act(self.dconv1(x))
        x2 = self.act(self.conv1(x1))
        return x2





class TNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[16, 32, 64, 128, 256, 512, 1024], act_mode='R'):
        super(TNet, self).__init__()

        # Convolutional layer without activation function
        self.head1 = B.conv(in_nc,nc[0], 3, mode='C')
        self.head2 = nn.Conv2d(nc[0], nc[0], 3, 2, 1, bias=True)
        self.head3 = nn.Conv2d(nc[0], nc[0], 3, 2, 1, bias=True)


        # Residual Dense Block
        self.rdb= RDB(nc=nc[0])

	# Rresidual Dense Block in BackBone
        self.rdb_bb1 = RDB(nc=nc[0])
        self.rdb_bb2 = RDB(nc=nc[1])
        self.rdb_bb3 = RDB(nc=nc[2])
        self.rdb_bb4 = RDB(nc=nc[3])
        self.rdb_bb5 = RDB(nc=nc[4])
        self.rdb_bb6 = RDB(nc=nc[5])


        # downsampling
        self.down1 = Downsampling(nc[0])
        self.down2 = Downsampling(nc[1])
        self.down3 = Downsampling(nc[2])
        self.down4 = Downsampling(nc[3])
        self.down5 = Downsampling(nc[4])
        self.down6 = Downsampling(nc[5])

        #  Dual attention module
        self.dual = DAM()

        # upsampling
        self.up1 = Upsampling(nc[6])
        self.up2 = Upsampling(nc[5])
        self.up3 = Upsampling(nc[4])
        self.up4 = Upsampling(nc[3])
        self.up5 = Upsampling(nc[2])
        self.up6 = Upsampling(nc[1])

        self.act = nn.ReLU(inplace=False)
	    #Convolutional layer
        self.tail = nn.Sequential(
            nn.Conv2d(nc[0],nc[0], kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(nc[0],nc[0], kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=nc[0], out_channels=out_nc, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, input_image):
        # ======================
        # naming convention description
        # Set the name of variables in order of applied block or convolutional layer
        # Ex) c1_r2 : means the result after one convolutional layer and two Residual Dense Block
        # In case of feature addition, adopt the result after RDB in backbone to reduce the length
        # c : Convolutional Layer
        # r : Residual Dense Block
        # d : Downsampling Block
        # u : Upsampling Block
        # f : Feature Addition
        # D : Dual Attention Module
        # ======================

        # Convolutional layer without activation function
        print(input_image.shape)
        c1 = self.head1(input_image)
        c1 = self.head2(c1)
        c1 = self.head3(c1)
        c1 = self.act(c1)
        print(c1.shape)
        # First three RDB
        c1_r1 = self.rdb(c1)
        c1_r2 = self.rdb(c1_r1)
        c1_r3 = self.rdb(c1_r2)
        print(c1_r3.shape)
        # Downsampling
        c1_r3_d1 = self.down1(c1_r3)
        print(c1_r3_d1.shape)
        c1_r3_d2 = self.down2(c1_r3_d1)
        print(c1_r3_d2.shape)
        c1_r3_d3 = self.down3(c1_r3_d2)
        print(c1_r3_d3.shape)
        c1_r3_d4 = self.down4(c1_r3_d3)
        print(c1_r3_d4.shape)
        c1_r3_d5 = self.down5(c1_r3_d4)
        print(c1_r3_d5.shape)
        c1_r3_d6 = self.down6(c1_r3_d5)
        print("here")
        print(c1_r3_d6.shape)
        # Dual attention module
        c1_r3_d6_D1 = self.dual(c1_r3_d6)

        # RDB in BackBone
        c1_r4 = self.rdb_bb1(c1_r3)
        c1_r3_d1_r1 = self.rdb_bb2(c1_r3_d1)
        c1_r3_d2_r1 = self.rdb_bb3(c1_r3_d2)
        c1_r3_d3_r1 = self.rdb_bb4(c1_r3_d3)
        c1_r3_d4_r1 = self.rdb_bb5(c1_r3_d4)
        c1_r3_d5_r1 = self.rdb_bb6(c1_r3_d5)

        # feature addition and upsampling
        c1_r3_d6_D1_u1 = self.up1(c1_r3_d6_D1)
        c1_r3_d5_r1_f1 = c1_r3_d5_r1 + c1_r3_d6_D1_u1
        c1_r3_d5_r1_f1_u1 = self.up2(c1_r3_d5_r1_f1)
        c1_r3_d4_r1_f1 = c1_r3_d4_r1 + c1_r3_d5_r1_f1_u1
        c1_r3_d4_r1_f1_u1 = self.up3(c1_r3_d4_r1_f1)
        c1_r3_d3_r1_f1 = c1_r3_d3_r1 + c1_r3_d4_r1_f1_u1
        c1_r3_d3_r1_f1_u1 = self.up4(c1_r3_d3_r1_f1)
        c1_r3_d2_r1_f1 = c1_r3_d2_r1 + c1_r3_d3_r1_f1_u1
        c1_r3_d2_r1_f1_u1 = self.up5(c1_r3_d2_r1_f1)
        c1_r3_d1_r1_f1 = c1_r3_d1_r1 + c1_r3_d2_r1_f1_u1
        c1_r3_d1_r1_f1_u1 = self.up6(c1_r3_d1_r1_f1)
        c1_r4_f1 = c1_r4 + c1_r3_d1_r1_f1_u1

        # last three RDB
        c1_r4_f1_r1 = self.rdb(c1_r4_f1)
        c1_r4_f1_r2 = self.rdb(c1_r4_f1_r1)
        c1_r4_f1_r3 = self.rdb(c1_r4_f1_r2)

        x0_h, x0_w = input_image.size(2), input_image.size(3)
        c1_r4_f1_r3 = F.interpolate(c1_r4_f1_r3, size=(x0_h, x0_w), mode='bilinear', align_corners=None)

        # Convolutional layer
        output_image = self.tail(c1_r4_f1_r3)      #output_image : c1_r4_f1_r3_c1
        return output_image
