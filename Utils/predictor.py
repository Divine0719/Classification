from pathlib import Path
import torch

from tqdm import tqdm


class Predictor(object):
    cuda = torch.cuda.is_available()

    def __init__(self, model, model_path):
        self.model = model
        if not model_path == None:
            self.test_acc_PATH=model_path
            self.test_acc_PATH=self.test_acc_PATH.rsplit('/',1)[0].rsplit('/',1)[0]
            model.load_state_dict(torch.load(model_path))
            print('loading checkpoint!        ' + model_path)
        else:
            print('checkpoint needed')
        if self.cuda:
            model.cuda()

    def _iteration(self, data_loader):
        accuracy = []
        for data, target in tqdm(data_loader, ncols=80):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
                data = data.to(device='cuda', dtype=torch.float32)
                target = target.to(device='cuda', dtype=torch.float32)
            output = self.model(data)
            accuracy.append((output.data.max(1)[1] == target.data.max(1)[1]).sum().item())
        print(">>>[{}]accuracy: {:.4f}".format('test', sum(accuracy) / len(data_loader.dataset)))
        return sum(accuracy) / len(data_loader.dataset)

    def test(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            correct = self._iteration(data_loader)

            with open(self.test_acc_PATH+'/'+"test_accuracy"+str(correct)+".txt", "a", encoding="utf-8") as f:
                f.write(str({"accuracy": correct}))
                f.write("\n")
            return correct


    def loop(self,test_data):
        acc=self.test(test_data)
        print(acc)
