import torchvision.models as models
import torch.nn as nn
# import torchvision.models.ResNet50_Weights

def resnet18(num_classes=1000):
    model = models.resnet18(pretrained=True)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet50(num_classes=1000):
    model = models.resnet50(pretrained=True)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet101(num_classes=1000):
    model = models.resnet101(pretrained=True)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet152(num_classes=1000):
    model = models.resnet152(pretrained=True)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def vgg16(num_classes=1000):
    model = models.vgg16(pretrained=True)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# resnet18 = models.resnet18(pretrained=True)
# alexnet = models.alexnet(pretrained=True)
# squeezenet = models.squeezenet1_0(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)
# densenet = models.densenet161(pretrained=True)
# inception = models.inception_v3(pretrained=True)
# googlenet = models.googlenet(pretrained=True)
# shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
# mobilenet_v2 = models.mobilenet_v2(pretrained=True)
# mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
# mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
# resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
# wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
# mnasnet = models.mnasnet1_0(pretrained=True)
# efficientnet_b0 = models.efficientnet_b0(pretrained=True)
# efficientnet_b1 = models.efficientnet_b1(pretrained=True)
# efficientnet_b2 = models.efficientnet_b2(pretrained=True)
# efficientnet_b3 = models.efficientnet_b3(pretrained=True)
# efficientnet_b4 = models.efficientnet_b4(pretrained=True)
# efficientnet_b5 = models.efficientnet_b5(pretrained=True)
# efficientnet_b6 = models.efficientnet_b6(pretrained=True)
# efficientnet_b7 = models.efficientnet_b7(pretrained=True)
# regnet_y_400mf = models.regnet_y_400mf(pretrained=True)
# regnet_y_800mf = models.regnet_y_800mf(pretrained=True)
# regnet_y_1_6gf = models.regnet_y_1_6gf(pretrained=True)
# regnet_y_3_2gf = models.regnet_y_3_2gf(pretrained=True)
# regnet_y_8gf = models.regnet_y_8gf(pretrained=True)
# regnet_y_16gf = models.regnet_y_16gf(pretrained=True)
# regnet_y_32gf = models.regnet_y_32gf(pretrained=True)
# regnet_x_400mf = models.regnet_x_400mf(pretrained=True)
# regnet_x_800mf = models.regnet_x_800mf(pretrained=True)
# regnet_x_1_6gf = models.regnet_x_1_6gf(pretrained=True)
# regnet_x_3_2gf = models.regnet_x_3_2gf(pretrained=True)
# regnet_x_8gf = models.regnet_x_8gf(pretrained=True)
# regnet_x_16gf = models.regnet_x_16gf(pretrainedTrue)
# regnet_x_32gf = models.regnet_x_32gf(pretrained=True)