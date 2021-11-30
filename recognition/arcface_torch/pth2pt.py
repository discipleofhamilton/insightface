import sys, os

import torch 
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models

from backbones import get_model

import numpy as np
import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--pth_model', type=str, help="Set path to (input) model(.pth)", default=None)
    parser.add_argument('-o', '--pt_model_dir', type=str, help='Set directory path to (output) model(.pt)', default=None)
    parser.add_argument('-n', '--network', type=str, default=None, help='backbone network')
    return parser.parse_args(argv)

def check_model_name(model, mode):

    model_name, model_subname = model.split(".")

    # input model: check .pth
    if mode == 0:
        if model_subname != "pth":
            print("\tERROR: INVALID INPUT MODEL NAME!!!")
            return False
    
    # output model: check .pt
    elif mode == 1:
        if model_subname != "pt":
            print("\tERROR: INVALID INPUT MODEL NAME!!!")
            return False

    else:
        print("\tERROR: INVALID MODE OF CHECKING ODEL NAME!!!")
        return False

    return True

if __name__ == "__main__":

    argv = parse_arguments(sys.argv[1:])

    if argv.pth_model is None or not os.path.isfile(argv.pth_model) or not check_model_name(argv.pth_model, 0) :
        print("\tPlease set a valid pth model to be converted.")
        sys.exit()

    if argv.pt_model_dir is None:
        print("\tPlease set a directory to store converted model.")
        sys.exit()
    
    if not os.path.exists(argv.pt_model_dir):
        os.makedirs(argv.pt_model_dir)

        os.path.basename

    pth_model_name = os.path.basename(os.path.dirname(argv.pth_model)).lower()
    params = pth_model_name.split("_")
    if len(params) >= 3 and params[1] in ('arcface', 'cosface'):
        if argv.network is None:
            argv.network = params[2]
        
    # set device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set pt model
    model_name, model_subname = os.path.basename(argv.pth_model).split(".")
    pt_model_name = model_name + ".pt"
    pt_model = os.path.join(argv.pt_model_dir, pt_model_name) 

    print("pt_model", pt_model)

    print("argv.network", argv.network)

    # load pth model
    model = get_model(argv.network, dropout=0) # get backbone
    model = model.to(device)
    assert isinstance(model, torch.nn.Module)
    weight = torch.load(argv.pth_model) # get weight
    model.load_state_dict(weight)  # load weight
    model.eval() # switch to eval mode

    # convert pth to pt
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    # img.div_(255).sub_(0.5).div_(0.5)
    img = img.to(device)
    traced_script_module = torch.jit.trace(model, img)
    traced_script_module.save(pt_model)