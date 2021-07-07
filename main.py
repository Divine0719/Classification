from train import do_train
from predict import do_predict
import argparse

def main(mode,num_classes,data_type,model_type,model_index,batch_size,train_epoch,predict_epoch,is_transfer):
    if mode=='train':
        do_train(num_classes, data_type,model_type,model_index ,train_epoch, batch_size,is_transfer)
    elif mode=='predict':
        do_predict(num_classes, data_type,model_type,model_index ,predict_epoch, batch_size,is_transfer)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default='train')
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--data_type", type=str, default='Covid19_Pneumonia')
    p.add_argument("--model_type", type=int, default=2)#查看model_zoo
    p.add_argument("--model_index", type=int, default=3)#查看model_zoo
    p.add_argument("--batch_size", type=int, default=160)
    p.add_argument("--train_epoch", type=int, default=100)
    p.add_argument("--predict_epoch", type=int, default=1)  # 指的是使用第几个epoch的参数进行预测
    p.add_argument("--is_transfer", type=bool, default=True)#迁移训练一般和预训练pretrained挂钩,但是预测时与预训练模型参数无关

    args = p.parse_args()
    main(args.mode,args.num_classes,args.data_type,args.model_type,args.model_index,args.batch_size,args.train_epoch,args.predict_epoch,args.is_transfer)