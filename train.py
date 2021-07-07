from Models.baseline.VGG import vgg19
from Models.baseline.DenseNet import densenet121,densenet201
from  Models.baseline.MobileNetV3 import MobileNetV3_Large
from Utils.dataloader import get_dataloader
from Utils.trainer import Trainer
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import torch
from pathlib import Path
import glob
import os
from Models.model_zoo import get_model,get_model_name
def do_train(num_classes, data_type, model_type, model_index, epoch, batch_size, is_transfer):
    model_name=get_model_name(model_type,model_index)
    train_loader = get_dataloader(data_type,'train',batch_size,True,need_cvtBGR2Gray=False)
    val_loader = get_dataloader(data_type,'val',batch_size,True,need_cvtBGR2Gray=False)
    if not is_transfer:
        model = get_model(model_type,model_index,num_classes,is_transfer).cuda()
#########################################################################################
#                                                                                       #
#                                      模型迁移部分                                       #
#                                                                                       #
#########################################################################################
    if is_transfer:
        model = get_model(model_type,model_index,1000,is_transfer).cuda()#num_classes  如果不是迁移训练就设置为num_classes
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
        fc = nn.Sequential(
            nn.BatchNorm1d(1920),
            nn.Linear(1920, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
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

    # optimizer = optim.SGD(params=model.parameters(), lr=1e-2, momentum=0.9,
    #                       weight_decay=1e-4)
    optimizer=torch.optim.Adamax(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 3, 0.5)
    #获取最新的ckpt
    ckpt_path='./SaveFolder/' + data_type +'/' + model_name + '/Weights'
    ckpt_path=Path(ckpt_path)
    if ckpt_path.exists():
        files = os.listdir(ckpt_path)
        files_path = [f'{ckpt_path}\\{file}' for file in files]
        files_path.sort(key=lambda fp: os.path.getctime(fp), reverse=True)
        for index in range(len(files_path)):#排除ckpt以外格式的文件
            if 'ckpt' in files_path[index]:
                newest_ckpt = files_path[index]
                print(newest_ckpt)
                break
    else:
        newest_ckpt=None

    trainer = Trainer(model, optimizer, nn.CrossEntropyLoss(), save_dir="./SaveFolder/" + data_type +'/' + model_name,
                      checkpoint_PATH=newest_ckpt)#nn.BCEWithLogitsLoss()更适合多标签分类
    trainer.loop(epoch, train_loader, val_loader, scheduler)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--data_type", type=str, default='ASD')
    p.add_argument("--model_type",type=str,default='densenet201')
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--batchsize", type=int, default=16)
    p.add_argument("--is_transfer", type=bool, default=False)

    args = p.parse_args()
    do_train(args.num_classes, args.data_type, args.model_type, args.epoch,args.batchsize,args.is_transfer)
