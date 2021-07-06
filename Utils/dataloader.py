import matplotlib.pyplot as plt
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
"""
本函数主要实现数据加载，
数据集格式如
    data/
        数据集1/
            train/
                类别1/
                类别2/
            val/
                ..
            test/
                ..
        数据集2/
            train/
                类别1/
                类别2/
仅需要调用：
get_dataloader(data_type,data_usage,batch_size=16,suffle=True,need_cvtBGR2Gray=False)
例如：
isbi_dataset = get_dataloader('ASD','train',16,True,need_cvtBGR2Gray=False)
"""

class ISBI_Loader(Dataset):
    def __init__(self, data_type,data_usage,need_cvtBGR2Gray=False):
        # 初始化函数，读取所有data_path下的图片
        self.need_cvtBGR2Gray=need_cvtBGR2Gray
        self.data_path = '../data/'
        self.classes = glob.glob(os.path.join(self.data_path, data_type+'/'+data_usage+'/*'))
        self.num_class = len(self.classes)
        print(data_usage,'数据集类别数:',str(self.num_class))
        self.imgs_path = glob.glob(os.path.join(self.data_path, data_type+'/train/*/*.jpg'))
        print(data_usage,'数据集各类别图片总数：',str(len(self.imgs_path)))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
    #获取onehot码，输入为类别序号和类别总数(序号从0开始)，如输入(3,5),则返回[0,0,0,1,0]
    def get_onehot(self,class_index,num_class):
        onehot_code=[]
        for i in range (num_class):
            if not i==class_index:
                onehot_code.append(0)
            else:
                onehot_code.append(1)
        onehot_code=np.array(onehot_code)
        return onehot_code

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        # 必须将尺寸转为正方形，避免因此在flip过程中产生错误
        image = cv2.resize(image, (224, 224))
        #图片显示
        # cv2.imshow('',image)
        # cv2.waitKey()
        if self.need_cvtBGR2Gray:
            image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            # opencv读入的数据格式为HWC，将其改为CHW
            image = image.reshape(1, image.shape[0], image.shape[1])
        else:
            # opencv读入的数据格式为HWC，将其改为CHW
            image = image.reshape(3, image.shape[0], image.shape[1])
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:  # 确保image和label使用同一个flipcode
            image = self.augment(image, flipCode)
        for class_idx in range(len(self.classes)):
            if self.classes[class_idx] in image_path:
                label=self.get_onehot(class_idx,self.num_class)
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

def get_dataloader(data_type,data_usage,batch_size=16,suffle=True,need_cvtBGR2Gray=False):
    isbi_dataset = ISBI_Loader(data_type,data_usage,need_cvtBGR2Gray=need_cvtBGR2Gray)
    data_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=suffle)
    for image, label in data_loader:
        print(label)
        break
    return data_loader

if __name__ == "__main__":
    isbi_dataset = get_dataloader('ASD','train',16,True,need_cvtBGR2Gray=False)
