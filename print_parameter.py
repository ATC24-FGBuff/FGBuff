from torchsummary import summary
from models.vgg import vgg16_bn
from models.resnet import resnet50
from models.vgg import vgg19_bn

# model = vgg16_bn()
# model = resnet50()
# model = vgg19_bn()
# model = models.resnet50()
# model = models.vgg16_bn()
# summary(model.cuda(), (3,32,32))

from torchvision import models
# model = models.resnet101()
# summary(model.cuda(), (3,32,32))


model = models.resnet152()
summary(model.cuda(), (3,32,32))


# model = models.resnet18()
# summary(model.cuda(), (3,32,32))


# model = models.resnet50()
# summary(model.cuda(), (3,32,32))

