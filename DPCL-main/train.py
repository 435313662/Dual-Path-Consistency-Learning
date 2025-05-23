import os
import sys
import time
import glob
import numpy as np
import torch
import torch.optim.sgd
import utils
from PIL import Image
import logging
import argparse
import random
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from model import *
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("DPCL")
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=201, help='epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')
parser.add_argument('--save', type=str, default='EXP/', help='location of the data corpus')
parser.add_argument('--isRandom', type=bool, default=True, help='random')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main(train_low_data_names,val_low_data_names):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)


    model = Network(stage=args.stage,isRandom=args.isRandom)

    model.enhance.conv.apply(model.weights_init)
    model.enhance.out_conv.apply(model.weights_init)
    if args.isRandom==False:
        model.calibrate.in_conv.apply(model.weights_init)
        model.calibrate.convs.apply(model.weights_init)
        model.calibrate.out_conv.apply(model.weights_init)

    # model_dict = model.state_dict()
    # pretrained_dict = torch.load('EXP/best.pt')
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)


    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)

    
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')
    valDataset = MemoryFriendlyLoader(img_dir=val_low_data_names, task='test')
    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=True,generator=torch.Generator(device = 'cuda'))
    val_queue = torch.utils.data.DataLoader(
        valDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=True,generator=torch.Generator(device = 'cuda'))

    total_step = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_idx, (input, name) in enumerate(train_queue): 
            total_step += 1
            input = Variable(input, requires_grad=False).cuda()
            optimizer.zero_grad()
            loss = model._loss(input)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            losses.append(loss.item())
            logging.info('train-epoch-batch %03d %04d %f', epoch, batch_idx, loss)



        logging.info('train-epoch %03d %f', epoch, np.average(losses))
        utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))

        if epoch % 200 == 0 and total_step != 0:
            model.eval()
            with torch.no_grad():
                valsave_path=image_path +'/'+str(epoch)+'/'+'val'
                os.makedirs(valsave_path, exist_ok=True)
                for batch_idx, (input, image_name) in enumerate(val_queue):
                    input = Variable(input, volatile=True).cuda()
                    image_name = image_name[0].split('\\')[-1].split('.')[0]
                    illu_list, ref_list, input_list, atten= model(input)
                    u_name = '%s.png' % (image_name)
                    u_path = valsave_path+ '/' + u_name
                    save_images(ref_list[0], u_path)


if __name__ == '__main__':
    train_low_data_names = '/data'
    val_low_data_names = '/data'
    main(train_low_data_names,val_low_data_names)
