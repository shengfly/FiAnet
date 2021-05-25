#*****************************************************
#
#  This is the PyTorch code for our paper:
#  Multi-channel attention-fusion neural network for brain age estimation: Accuracy, generality, and interpretation with 16,705 healthy MRIs across lifespan
#  Medical Image Analysis, Volume 72, 2021, 102091
#
#  @email: heshengxgd@gmail.com
#
#*****************************************************

import torch
import torch.nn as nn

def conv3d(in_planes, out_planes, stride=1, kernel_size=3, groups=1, dilation=1):
    """3x3 convolution with padding"""
    padding = (kernel_size-1)//2
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation)
    
class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,conv_layer=conv3d):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #if groups != 1 or base_width != 64: 
            #raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_layer(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class first_conv(nn.Module):
    def __init__(self,inplace,outplace,norm_layer,conv_layer):
        super().__init__()
        self.conv = conv_layer(inplace,outplace,kernel_size=1)
        self.bn = norm_layer(outplace)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
        
    
class Resnet(nn.Module):
    def __init__(self,num_classes=1,in_channels=1):
        super().__init__()
        
        self.bn3 = nn.InstanceNorm3d
        self.cv3 = conv3d
        
        self.dilation = 1
        self.groups = 1
        self.inplanes = 64
        self.base_width = 64
        
        layers = [2, 2, 2, 2]
        
        n_channel = [64,128,256,512]
        
        self.conv3d_1 = first_conv(in_channels,outplace=self.inplanes,norm_layer=self.bn3,conv_layer=self.cv3)
        self.conv3d_layer1 = self._make_layer(BasicBlock, n_channel[0], layers[0],norm_layer=self.bn3,conv_layer=self.cv3)
        self.conv3d_layer2 = self._make_layer(BasicBlock, n_channel[1], layers[1],stride=2,norm_layer=self.bn3,conv_layer=self.cv3)
        self.conv3d_layer3 = self._make_layer(BasicBlock, n_channel[2], layers[2],stride=2,norm_layer=self.bn3,conv_layer=self.cv3)
        self.conv3d_layer4 = self._make_layer(BasicBlock, n_channel[3], layers[3],stride=2,norm_layer=self.bn3,conv_layer=self.cv3)
        
        self.avgpool_3d = nn.AdaptiveAvgPool3d(1)
        self.regression_3d = nn.Linear(n_channel[3],num_classes)
    
    def _make_layer(self, block, planes, blocks, norm_layer, conv_layer,stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_layer(self.inplanes, planes * block.expansion, stride,kernel_size=1),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,conv_layer=conv_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,conv_layer=conv_layer))

        return nn.Sequential(*layers)
        
    def forward(self,x):
        
        x3d_0 = self.conv3d_1(x)
        x3d_1 = self.conv3d_layer1(x3d_0)
        x3d_2 = self.conv3d_layer2(x3d_1)
        x3d_3 = self.conv3d_layer3(x3d_2)
        x3d_4 = self.conv3d_layer4(x3d_3)

        feat3d = self.avgpool_3d(x3d_4)
        feat3d = torch.flatten(feat3d, 1)
        
        o3d = self.regression_3d(feat3d)
        
        return x3d_1,x3d_2,x3d_3,x3d_4,o3d

class fusion(nn.Module):
	def __init__(self,inplace,outplace,first):
		super().__init__()

		self.first = first

		self.conv = nn.Conv3d(4 * inplace, outplace,3,1,1,bias=False)
		self.bn = nn.InstanceNorm3d(outplace)
		self.relu = nn.ReLU(inplace=True)
		if self.first:
			self.conv2 = nn.Conv3d(outplace, outplace,3,1,1,bias=False)
		else:
			self.conv2 = nn.Conv3d(outplace+inplace, outplace,3,1,1,bias=False)

		self.bn2 = nn.InstanceNorm3d(outplace)

		self.kernel_se_conv = nn.Conv3d(2*inplace,inplace,3,1,1)

		self.convs1 = nn.Conv3d(inplace,inplace,3,1,1,bias=False)
		self.convs2 = nn.Conv3d(inplace,inplace,3,1,1,bias=False)
		self.convm1 = nn.Conv3d(inplace,inplace,3,1,1,bias=False)
		self.convm2 = nn.Conv3d(inplace,inplace,3,1,1,bias=False)

		#self.softmax = nn.Softmax(dim=1)

	def forward(self,x1,x2,z=None):
		x = torch.cat([x1,x2],1)
		x = self.kernel_se_conv(x)
		x = torch.sigmoid(x)
		y1 = x * x1 + (1-x) * x2
		y2 = torch.sigmoid(self.convs1(x1)) * x2
		y3 = torch.sigmoid(self.convs2(x2)) * x1
		y4 = torch.max(self.convm1(x1),self.convm2(x2))

		y = torch.cat([y1,y2,y3,y4],1)
		y = self.relu(self.bn(self.conv(y)))
		if self.first:
			y = self.relu(self.bn2(self.conv2(y)))
		else:
			y = torch.cat([y,z],1)
			y = self.relu(self.bn2(self.conv2(y)))
			
		return y

class fusNet(nn.Module):
	def __init__(self,num_classes=1):
		super().__init__()

		self.net1 = Resnet(num_classes)
		self.net2 = Resnet(num_classes)
		
		self.fus1 = fusion(64,128,True)
		self.fus2 = fusion(128,256,False)
		self.fus3 = fusion(256,512,False)
		self.fus4 = fusion(512,512,False)
		

		self.maxp = nn.MaxPool3d(2,2)
		self.avgpool_3d = nn.AdaptiveAvgPool3d(1)
		self.regression_3d = nn.Linear(512,num_classes)

	def forward(self,ix1,ix2):
		x1,x2,x3,x4,f1 = self.net1(ix1)
		y1,y2,y3,y4,f2 = self.net2(ix2)

		z = self.fus1(x1,y1)
		z = self.maxp(z)
		z = self.fus2(x2,y2,z)
		z = self.maxp(z)
		z = self.fus3(x3,y3,z)
		z = self.maxp(z)
		z = self.fus4(x4,y4,z)
		z = self.maxp(z)

		z = self.avgpool_3d(z)
		z = torch.flatten(z,1)
		f3 = self.regression_3d(z)

		return f1,f2,f3

	

if __name__ == '__main__':
	x = torch.rand(1,1,64,64,64)
	y = torch.rand(1,1,64,64,64)
	mod = fusNet()
	f = mod(x,y)
	for i in f:
		print(i.shape)
