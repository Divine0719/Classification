from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm


class Trainer(object):
    cuda = torch.cuda.is_available()
    #torch.backends.cudnn.benchmark = True

    def __init__(self, model, optimizer, loss_f, save_dir=None, save_freq=5,checkpoint_PATH=None):
        self.model = model
        self.start_epoch =1
        if not checkpoint_PATH==None:
            checkpoint_PATH=checkpoint_PATH
            model_CKPT = torch.load(checkpoint_PATH)
            model.load_state_dict(model_CKPT['net_state_dict'])
            self.start_epoch = model_CKPT['epoch']+1
            print('loading checkpoint!        '+checkpoint_PATH)
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_f = loss_f
        self.save_dir = save_dir
        self.save_freq = save_freq

    def _iteration(self, data_loader, is_train=True):
        loop_loss = []
        accuracy = []
        for data, target in tqdm(data_loader, ncols=80):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
                data = data.to(device='cuda', dtype=torch.float32)
                target = target.to(device='cuda', dtype=torch.float32)
                target=target.max(1)[1]#交叉熵的计算中，标签需要从[0,0,0,1,0]转为3
            output = self.model(data)
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data.item() / len(data_loader))
            #accuracy.append((output.data.max(1)[1] == target.data.max(1)[1]).sum().item())
            accuracy.append((output.data.max(1)[1] == target.data).sum().item())#交叉熵的计算中，标签需要从[0,0,0,1,0]转为3,因此准确率计算式也得转化
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print()
        mode = "train" if is_train else "val"
        print(">>>[{}] loss: {:.4f}/accuracy: {:.4f}".format(mode, sum(loop_loss),
                                                             sum(accuracy) / len(data_loader.dataset)))
        return mode, sum(loop_loss), sum(accuracy) / len(data_loader.dataset)

    def train(self, data_loader):
        self.model.train()
        with torch.enable_grad():
            mode, loss, correct = self._iteration(data_loader)
            return mode, loss, correct

    def val(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            mode, loss, correct = self._iteration(data_loader, is_train=False)
            return mode, loss, correct

    def loop(self, epochs, train_data, val_data, scheduler=None):
        for ep in range(self.start_epoch, epochs + 1):
            print("epochs: {}".format(ep))
            self.train(train_data)
            self.val(val_data)
            if scheduler is not None:
                scheduler.step()
            #save statistics into txt file
            self.save_statistic(*((ep,) + self.train(train_data)))
            self.save_statistic(*((ep,) + self.val(val_data)))
            if ep % self.save_freq:
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir+'/Weights')
            state = {"epoch": epoch, "net_state_dict": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir(exist_ok=True, parents=True) # 递归创建文件目录
            torch.save(state, model_out_path / "model_epoch_{}.ckpt".format(epoch))

    def save_statistic(self, epoch, mode, loss, accuracy):
        state_out_path = Path(self.save_dir)
        if not state_out_path.exists():
            state_out_path.mkdir(exist_ok=True, parents=True)  # 递归创建文件目录
        with open(self.save_dir+"/state_"+mode+".csv", "a", encoding="utf-8") as f:
            f.write('epoch,'+str(epoch)+',')
            f.write('mode,'+str(mode)+',')
            f.write('loss,'+str(loss)+',')
            f.write('accuracy,'+str(accuracy))
            f.write("\n")
