import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Res2Net_v1b import res2net50_v1b_26w_4s
from models.modules import  ASM,NonLocalBlock,SELayer
class edge_Block(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=1):
        super(edge_Block, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel,3,stride=1,padding=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(2*in_channel, out_channel,3,stride=1,padding=1),  # conv2d  -》 bn -》relu
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel,3,stride=1,padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel,3,stride=1,padding=1),
        )
        self.se = SELayer(128)

    def forward(self, x,gc):
        x0 = self.branch0(x)
        gc_channel = torch.sigmoid(gc) * x
        gc_channel = self.se(gc_channel)
        x0 =  torch.cat((x0,gc_channel),dim=1)     
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        out = x3 + x
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class RFB_modified1_4(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified1_4, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
    def forward(self, x):
        x0 = self.branch0(x)
        return x0

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel,l):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),   # conv2d  -》 bn -》relu
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3*(5-l), dilation=3*(5-l))
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5*(5-l), dilation=5*(5-l))
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7*(5-l), dilation=7*(5-l))
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.non_local = NonLocalBlock(in_channel)
    def forward(self, x):
        x0 = self.branch0(x)
        x0 = self.non_local(x0)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x
class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_ms1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_ms2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv5 = nn.Conv2d(channel, 1, 1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x1, x2, x3):
        x1 = self.conv_upsample1(self.upsample(self.upsample(x1)))
        x2 = self.conv_upsample2(self.upsample(x2))
        x3_x1 = x3-x1
        x3_x1 = self.conv_ms1(x3_x1)
        x2_x1 = x2-x1
        x2_x1 = self.conv_ms2(x2_x1)
        x = x3_x1 + x2_x1 +x1
        x = self.conv4(x)
        gc_channel = x
        x = self.conv5(x)
        x = self.pool(self.pool(x))
        gc_channel = self.pool(gc_channel)
        return x,gc_channel

class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(DecoderBlock, self).__init__()
        self.conv1 = BasicConv2d(in_channels,in_channels // 2,kernel_size=kernel_size,
                                 stride=stride , padding=padding)
        self.conv2 =BasicConv2d(in_channels // 2,out_channels,kernel_size=kernel_size,
                                 stride=stride , padding=padding)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class LCA(nn.Module):         # 粗粒度残差学习
    def __init__(self):
        super(LCA, self).__init__()
        self.conv1 = BasicConv2d(128,128,3,1,1)
        self.conv2 = BasicConv2d(256, 128, 3, 1, 1)
        self.se = SELayer(256)
    def forward(self, en_map, pred,gc):
        residual = en_map
        att = 1 - pred
        att_x =en_map * att
        att_y = torch.sigmoid(gc) * en_map
        badr = 1-abs(pred-0.5)/0.5
        att_z = badr*en_map
        out = att_x + att_y
        out = self.se(torch.cat((out,att_z),dim=1))
        out = self.conv2(out)
        out = out + en_map
        out = self.conv1(out)
        return out,att_z

class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = BasicConv2d(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.dropout = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(in_channels //4,out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class JEGNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=128):
        super(ACSNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True) #返回Res2net模型
        # ---- Receptive Field Block like module ----
        self.rfb0_1 = RFB_modified1_4(64, channel)
        self.rfb1_1 = RFB_modified1_4(256, channel)
        self.rfb2_1 = RFB_modified1_4(512, channel)
        self.rgb2_gloab = RFB_modified(128,channel,2)
        self.rfb3_1 = RFB_modified1_4(1024, channel)
        self.rgb3_gloab =RFB_modified(128,channel,3)
        self.rfb4_1 = RFB_modified1_4(2048, channel)
        self.rfb4_gloab = RFB_modified(128, channel,4)
        #lca
        self.lca0=LCA()
        self.lca1=LCA()
        self.lca2=LCA()
        self.lca3=LCA()
        # Sideout
        self.sideout0 = SideoutBlock(128, 1)
        self.sideout1 = SideoutBlock(128, 1)
        self.sideout2 = SideoutBlock(128, 1)
        self.sideout3 = SideoutBlock(128, 1)
        self.sideout4 = SideoutBlock(128, 1)
        # Decoder
        self.decoder0 = DecoderBlock(in_channels=384, out_channels=128)
        self.decoder1 = DecoderBlock(in_channels=384, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=384, out_channels=128)
        self.decoder3 = DecoderBlock(in_channels=384, out_channels=128)
        self.decoder4 = DecoderBlock(in_channels=128, out_channels=128)
        #SEmoudle
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.agg1 = aggregation(channel)
        self.asm0 =ASM(128,384)
        self.asm1 =ASM(128,384)
        self.asm2 =ASM(128,384)
        self.asm3 =ASM(128,384)
        self.edge_en1 = edge_Block(128,128)
        self.edge_en2_1 = edge_Block(128,128)
        self.sideout_edge1 = SideoutBlock(128, 1)
        self.sideout_edge2 = SideoutBlock(128, 1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = self.resnet.conv1(x) #经过三次卷积 # bs,64,176,176
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)        #bs ,64,176,176
        x0 = self.resnet.maxpool(x)        #bs ,64,88,88
        x1 = self.resnet.layer1(x0)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
        x0_rfb = self.rfb0_1(x)        # channel -> 128
        x1_rfb = self.rfb1_1(x1)        # channel -> 128
        x2_rfb = self.rfb2_1(x2)        # channel -> 128     
        x2_gloab = self.rgb2_gloab(x2_rfb)
        x3_rfb = self.rfb3_1(x3)        # channel -> 128
        x3_gloab =self.rgb3_gloab(x3_rfb)
        x4_rfb = self.rfb4_1(x4)        # channel -> 128
        x4_gloab = self.rfb4_gloab(x4_rfb)        # channel -> 128
        gc,gc3_channel = self.agg1(x4_gloab,x3_gloab,x2_gloab)
        gc3 = self.up(gc)  #gc3 22*22
        gc2 = self.up(gc3)#gc2 44*44
        gc1 = self.up(gc2)#gc1 88*88
        gc0 = self.up(gc1)  # 176*176
        gc2_channel = self.up(gc3_channel)
        x0_rfb_edge1 = self.edge_en1(x0_rfb,gc0)
        edge0 = self.sideout_edge1(x0_rfb_edge1)
        x1_rfb_edge1 = self.edge_en2_1(x1_rfb,gc1)
        edge1 = self.sideout_edge2(x1_rfb_edge1)
        d4 = self.decoder4(x4_rfb) # 128* 22 *22
        pred4 = self.sideout3(d4)
        merge3,attx_y3 = self.lca3(x3_rfb,torch.sigmoid(pred4),gc3)
        d3 = self.decoder3(self.asm3(merge3,d4,self.pool(self.pool(torch.sigmoid(edge1)))*x3_rfb+x3_rfb  ))
        pred3 = self.sideout2(d3)
        merge2,attx_y2= self.lca2(x2_rfb,torch.sigmoid(pred3),gc2)
        d2 = self.decoder2(self.asm2(merge2,d3,self.pool(torch.sigmoid(edge1))*x2_rfb+x2_rfb ))
        pred2 = self.sideout1(d2)
        merge1,attx_y1 = self.lca1(x1_rfb,torch.sigmoid(pred2),gc1)
        d1 =self.decoder1(self.asm1(merge1,d2,torch.sigmoid(edge1)*x1_rfb+x1_rfb ))
        pred1 =self.sideout0(d1)
        merge0 ,attx_y0= self.lca0(x0_rfb,torch.sigmoid(pred1),gc0)
        d0 = self.decoder0(self.asm0(merge0,d1,torch.sigmoid(edge0)*x0_rfb+x0_rfb ))
        pred0=self.sideout4(d0)
        return torch.sigmoid(pred0),torch.sigmoid(pred1),torch.sigmoid(pred2),torch.sigmoid(pred3),torch.sigmoid(pred4),\
        torch.sigmoid(edge1),torch.sigmoid(edge0),torch.sigmoid(gc),torch.sigmoid(merge3)




if __name__ == '__main__':
    import torch
    import torchvision
    from thop import profile
    import time
    from tqdm import tqdm

    print('==> Building model..')

    ras = ACSNet().cuda()
    input = torch.randn(1, 3, 320, 320)
    flops, params = profile(ras, (input.cuda(),))
    print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
    print('######################')

    # nelement()：统计Tensor的元素个数
    # .parameters()：生成器，迭代的返回模型所有可学习的参数，生成Tensor类型的数据
    total = sum([param.nelement() for param in ras.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
