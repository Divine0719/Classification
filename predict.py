from Models.baseline.VGG import vgg19
from Models.baseline.DenseNet import densenet201,densenet121
from  Models.baseline.MobileNetV3 import MobileNetV3_Large
from Utils.dataloader import get_dataloader
from Utils.predictor import Predictor
import glob
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import os
from pathlib import Path
from Utils.ckpt2pth import ckpt2pth
from Models.model_zoo import get_model,get_model_name

def do_predict(num_classes, data_type, model_type, model_index, epoch, batch_size, is_transfer=False):
    model_name=get_model_name(model_type,model_index)
    test_loader = get_dataloader(data_type,'test',batch_size,True,need_cvtBGR2Gray=False)
    #model = VGG19(num_classes=2).cuda()
    model=get_model(model_type,model_index,num_classes,pretrained=False).cuda()#必定加载非预训练模型
    #########################################################################################
    #                                                                                       #
    #                                      模型迁移部分                                       #
    #                                                                                       #
    #########################################################################################
    if is_transfer:
        fc = nn.Sequential(
            nn.BatchNorm1d(1920),
            nn.Linear(1920, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Softmax()
        )
        model.classifier = fc
        model.cuda()
    #########################################################################################
    #                                                                                       #
    #                                      模型迁移部分                                       #
    #                                                                                       #
    #########################################################################################
    ckpt_path='./SaveFolder/' + data_type +'/' + model_name + '/Weights'
    ckpt_file = ckpt_path+'/model_epoch_'+str(epoch)+'.ckpt'
    ckpt2pth(model,ckpt_file)#从ckpt转成pth再使用，避免出错
    predictor = Predictor(model, model_path=ckpt_file.replace('ckpt','pth'))
    predictor.loop(test_loader)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--data_type", type=str, default='ASD')
    p.add_argument("--model_type",type=str,default='densenet201')
    p.add_argument("--epoch", type=int, default=27)
    p.add_argument("--batchsize", type=int, default=24)
    p.add_argument("--is_transfer", type=bool, default=False)
    args = p.parse_args()
    do_predict(args.num_classes, args.data_type, args.model_type,args.epoch, args.batchsize,args.is_transfer)
