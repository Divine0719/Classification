from Models.baseline.VGG import vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn
from Models.baseline.MobileNetV3 import MobileNetV3_Small,MobileNetV3_Large
from Models.baseline.DenseNet import densenet121,densenet161,densenet169,densenet201

#模型数组
vgg=[vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn]
MobileNet=[MobileNetV3_Small,MobileNetV3_Large]
densenet=[densenet121,densenet161,densenet169,densenet201]
#模型名字数组
vgg_name=['vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn']
MobileNet_name=['MobileNetV3_Small','MobileNetV3_Large']
densenet_name=['densenet121','densenet161','densenet169','densenet201']

#model_zoo
model_zoo=[]
model_zoo.append(vgg)
model_zoo.append(MobileNet)
model_zoo.append(densenet)

#model_name_zoo
model_name_zoo=[]
model_name_zoo.append(vgg_name)
model_name_zoo.append(MobileNet_name)
model_name_zoo.append(densenet_name)



def get_model(model_type,model_index,num_classes,pretrained=False):
    if pretrained==False:
        return model_zoo[model_type][model_index](num_classes=num_classes)
    else:
        return model_zoo[model_type][model_index](num_classes=1000,pretrained=True)

def get_model_name(model_type,model_index):
    return model_name_zoo[model_type][model_index]

if __name__ == '__main__':
    model=get_model(2,0,2,False)
    print(model)