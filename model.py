import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

# #####################################################################
#kaiming初始化
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

#分类器初始化
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
#特征提取器
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)] 
        feat_block += [nn.BatchNorm1d(low_dim)]
        
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x

#分类器
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []       
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        x = self.classifier(x)
        return x       

# Define the ResNet18-based Model
class visible_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18'):
        super(visible_net_resnet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.visible = model_ft
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        #前向传播：卷积块+BN+RELU+最大池化+4个层
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x
        
class thermal_net_resnet(nn.Module):
    def __init__(self, arch ='resnet18'):
        super(thermal_net_resnet, self).__init__()
        if arch =='resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch =='resnet50':
            model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.thermal = model_ft
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x

class base_resnet_v(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet_v, self).__init__()

        model_base = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

        for mo in model_base.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        num_part = 6
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part - 1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        # x = self.visible.avgpool(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        # x = self.dropout(x)
        return x

class base_resnet_t(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet_t, self).__init__()

        model_base = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

        for mo in model_base.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        num_part = 6
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part - 1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        # x = self.visible.avgpool(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        # x = self.dropout(x)
        return x

#part嵌入
class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop = 0.5, arch ='resnet50',  gm_pool='on'):
        super(embed_net, self).__init__()
        if arch =='resnet18':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            self.basenet_v = base_resnet_v(arch= arch)
            self.basenet_t = base_resnet_t(arch=arch)
            pool_dim = 512
        elif arch =='resnet50':
            self.visible_net = visible_net_resnet(arch = arch)
            self.thermal_net = thermal_net_resnet(arch = arch)
            self.basenet_v = base_resnet_v(arch=arch)
            self.basenet_t = base_resnet_t(arch=arch)
            pool_dim = 2048

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.gm_pool = gm_pool

        #特征由FeatureBlock提取
        #分类器是ClassBlock
        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature3 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature4 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature5 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature6 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.classifier1 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout = drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier6 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier = ClassBlock(low_dim, class_num, dropout=drop)
        self.bn = FeatureBlock(pool_dim, low_dim, dropout = drop)
        self.l2norm = Normalize(2)
        # self.bn = nn.BatchNorm1d(pool_dim)

        #添加之后的
        #都是为了传递更多有效的模态信息
        self.channel = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=258,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(258),
            nn.ReLU()
        )
        self.W_channel = nn.Sequential(
            nn.Conv2d(in_channels=259, out_channels=32,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1)
        )

        self.emb1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=5, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )
        self.emb2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=5, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )
        self.em3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=5, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )
    def forward(self, x1, x2, modal = 0 ):

        if modal==0:
            #modal为0：上分支为可见光模态，下分支为红外模态
            #x1是可见光图像的特征映射
            x1 = self.visible_net(x1)
            x1 = self.basenet_v(x1)
            # chunk方法可以对张量分块，返回一个张量列表
            #沿2轴分为6块
            x1 = x1.chunk(6,2)
            #将特征分为6个条纹
            #pytorch contiguous一般与transpose，permute,view搭配使用：使用transpose或permute进行维度变换后，调用contiguous,然后方可使用view对维度进行变形
            x1_0 = x1[0].contiguous().view(x1[0].size(0),-1)
            x1_1 = x1[1].contiguous().view(x1[1].size(0), -1)
            x1_2 = x1[2].contiguous().view(x1[2].size(0), -1)
            x1_3 = x1[3].contiguous().view(x1[3].size(0), -1)
            x1_4 = x1[4].contiguous().view(x1[4].size(0), -1)
            x1_5 = x1[5].contiguous().view(x1[5].size(0), -1)
            #x2是红外图像的特征映射
            x2 = self.thermal_net(x2)
            x2 = self.basenet_t(x2)
            x2 = x2.chunk(6, 2)
            x2_0 = x2[0].contiguous().view(x2[0].size(0), -1)
            x2_1 = x2[1].contiguous().view(x2[1].size(0), -1)
            x2_2 = x2[2].contiguous().view(x2[2].size(0), -1)
            x2_3 = x2[3].contiguous().view(x2[3].size(0), -1)
            x2_4 = x2[4].contiguous().view(x2[4].size(0), -1)
            x2_5 = x2[5].contiguous().view(x2[5].size(0), -1)
            #将特征进行连接
            x_0 = torch.cat((x1_0, x2_0), 0)
            x_1 = torch.cat((x1_1, x2_1), 0)
            x_2 = torch.cat((x1_2, x2_2), 0)
            x_3 = torch.cat((x1_3, x2_3), 0)
            x_4 = torch.cat((x1_4, x2_4), 0)
            x_5 = torch.cat((x1_5, x2_5), 0)
        elif modal ==1:
            #modal为1表示是可见模态
            x = self.visible_net(x1)
            x = self.basenet_v(x)
            x = x.chunk(6,2)
            x_0 = x[0].contiguous().view(x[0].size(0), -1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)
        elif modal ==2:
            #modal为2表示是红外模态
            x = self.thermal_net(x2)
            x = self.basenet_t(x)
            x = x.chunk(6, 2)
            x_0 = x[0].contiguous().view(x[0].size(0), -1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)

        # 将6个局部特征进行cat
        x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5), 1)
        b, c = x.size()
        c = c / 6
        c = int(c)
        xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
        #经过嵌入
        x_em1 = self.emb1(xc).squeeze(-1).sum(dim=1)
        x_em2 = self.emb2(xc).squeeze(-1).sum(dim=1)

        #两个嵌入特征之间的距离
        x_s = x_em1[0].expand(1, c) - x_em2[0].expand(1, c).t()
        x_s = x_s.unsqueeze(0)
        #计算FR中每个特征向量以及FI中所有特征向量间成对的距离
        for i in range(1, b):
            x_t = x_em1[i].expand(1, c) - x_em2[i].expand(1, c).t()
            x_t = x_t.unsqueeze(0)
            #关系矩阵
            x_s = torch.cat((x_s, x_t), 0)
        #经过一个嵌入
        x_sj = self.channel(x_s.unsqueeze(-1))
        #变形后的原特征经过一个嵌入
        x_t = self.em3(xc)
        x_t = torch.mean(x_t, dim=1, keepdim=True)
        #将原特征和关系特征融合
        x_t = torch.cat((x_sj, x_t), 1)
        #嵌入
        W_yc = self.W_channel(x_t).transpose(1, 2)
        x_o = F.sigmoid(W_yc)

        feat = self.avgpool(x_o)
        feat = feat.view(feat.size(0), -1)
        feat_bn = self.bn(feat)
        # print(feat_bn.shape)
        out = self.classifier(feat_bn)

        #BN
        y_0 = self.feature1(x_0)
        y_1 = self.feature2(x_1)
        y_2 = self.feature3(x_2)
        y_3 = self.feature4(x_3)
        y_4 = self.feature5(x_4)
        y_5 = self.feature6(x_5)

        #对特征y进行分类
        # print(y_0.shape)
        # y0: 64,512
        out_0 = self.classifier1(y_0)
        out_1 = self.classifier2(y_1)
        out_2 = self.classifier3(y_2)
        out_3 = self.classifier4(y_3)
        out_4 = self.classifier5(y_4)
        out_5 = self.classifier6(y_5)

        if self.training:
            return (out_0, out_1, out_2, out_3, out_4, out_5), (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4), self.l2norm(y_5)), self.l2norm(feat_bn), out
        else:
            x_0 = self.l2norm(x_0)
            x_1 = self.l2norm(x_1)
            x_2 = self.l2norm(x_2)
            x_3 = self.l2norm(x_3)
            x_4 = self.l2norm(x_4)
            x_5 = self.l2norm(x_5)
            x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5), 1)
            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)
            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            y_4 = self.l2norm(y_4)
            y_5 = self.l2norm(y_5)
            y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5), 1)
            return x, y

