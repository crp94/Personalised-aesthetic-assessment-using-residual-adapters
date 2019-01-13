from __future__ import print_function, division
import torch
import numpy as np
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from torch.autograd import Variable
from torchvision import models
from torch import nn
use_gpu = True
import argparse
import math


if use_gpu:
    cuda = torch.device('cuda:0')     # Default CUDA device
torch.cuda.set_device(cuda.index)


class BaselineModel(nn.Module):
    def __init__(self, num_classes, inputsize, keep_probability=0.5):

        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(inputsize, 256)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.bn2(out)
        out = self.fc3(out)
        return out

class convNet(nn.Module):
    #constructor
    def __init__(self,resnet,mynet):
        super(convNet, self).__init__()
        #defining layers in convnet
        self.resnet=resnet
        self.myNet=mynet
    def forward(self, x):
        x=self.resnet(x)
        x=self.myNet(x)
        return x


class convNet2(nn.Module):
    #constructor
    def __init__(self,resnet,mynet):
        super(convNet2, self).__init__()
        #defining layers in convnet
        self.avgpl=nn.AdaptiveAvgPool2d((224,224))
        self.rsn=resnet
        self.myNet=mynet
    def forward(self, x):
        x=self.avgpl(x)
        x=self.rsn(x)
        return x

def to_grayscale(image):
    """
    input is (d,w,h)
    converts 3D image tensor to grayscale images corresponding to each channel
    """
    image = torch.sum(image, dim=0)
    image = torch.div(image, image.shape[0])
    return image

def normalize(image):
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
    ])
    image = Variable(preprocess(image).unsqueeze(0))
    return image

def deprocess(image):
    return image * torch.Tensor([0.229, 0.224, 0.225]).cuda()  + torch.Tensor([0.485, 0.456, 0.406]).cuda()

def load_image(path):
    image = Image.open(path).convert('RGB')
    return image

mean=np.asarray([0.485, 0.456, 0.406])
std=np.asarray([0.229, 0.224, 0.225])

parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default=0.4, help='Intensity of the enhancement performed to the input picture')
parser.add_argument('--network', type=str, default='fine_tuned_flickerAES_normalized_dropout_resnet18_customnetworkadamnormalized.pt', help='path to the pretrained aesthetics prediction network')
parser.add_argument('--inputimage', type=str, help='Path for the input image')
parser.add_argument('--outputimage', type=str, default='output.jpg', help='Desired path for the output image')
arguments = parser.parse_args()

epsilon=arguments.epsilon
network=arguments.network
inputimage=arguments.inputimage
outputimage = arguments.outputimage



model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.out_features
net1 = BaselineModel(1, num_ftrs)
net_2=torch.load(network)
cnvnet2=convNet2(resnet=net_2, mynet=net1.cuda())

modulelist = list(cnvnet2.modules())
image = load_image(inputimage)
image_2 = normalize(image)
img_variable = Variable(image_2.cuda(), requires_grad=True) #convert tensor into a variable
output = cnvnet2.forward(img_variable)[0]
output =Variable(output, requires_grad=True)

target = Variable(torch.FloatTensor([2]).cuda(), requires_grad=False)
loss = torch.nn.MSELoss()
loss_cal = loss(output, target)
loss_cal.backward(retain_graph=True)

eps = epsilon
x_grad = img_variable
x_adversarial = img_variable.data + eps * x_grad          #find adv example
output_adv = cnvnet2.forward(Variable(x_adversarial))   #perform a forward pass on adv example

x=img_variable.cpu().detach().numpy()
x_adv=x_adversarial
x_adv = x_adv.squeeze(0)
x_adv = x_adv.mul(torch.FloatTensor(std).cuda().view(3,1,1)).add(torch.FloatTensor(mean).cuda().view(3,1,1)).cpu().detach().numpy()#reverse of normalization op
x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
x_adv = np.clip(x_adv, 0, 1)

result = Image.fromarray((x_adv * 255).astype(np.uint8))
result.save(outputimage)

print('Image successfully saved to ' + outputimage)
