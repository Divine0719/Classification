import torch
from pathlib import Path
def ckpt2pth(model, ckpt_file, save_pth_file=True, save_ptl_file=False):
    if not ckpt_file == None:
        model_CKPT = torch.load(ckpt_file)
        model.load_state_dict(model_CKPT['net_state_dict'])
        if not save_pth_file == False:#模型参数，默认保存
            torch.save(model.state_dict(), ckpt_file.replace('ckpt','pth'))
            print('pth saved')
        if not save_ptl_file == False:#整个模型，默认不存
            torch.save(model, ckpt_file.replace('ckpt','ptl'))
            print('ptl saved')
