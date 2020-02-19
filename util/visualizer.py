import importlib
import ntpath
import os
import time

import numpy as np
import torch
from torchvision.utils import make_grid

from . import util


class Visualizer():
    def __init__(self, opt, mode='train'):
        # self.opt = opt
        self.tf_log = opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        if self.tf_log:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'logs')
            self.writer = importlib.import_module('tensorboardX').SummaryWriter(self.log_dir)
            self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
            util.mkdirs([self.img_dir])
        else:
            if mode == 'train':
                self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
                util.mkdirs([self.img_dir])
            else:
                self.img_dir = os.path.join(opt.results_dir, opt.name, 'images')
                util.mkdirs([self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def ten2imggrid(self, ten, norm=True, scale_each=False, range=(-1., 1.), padding=0, nrow=8):
        if range != None:
            ten_grid = make_grid(ten, normalize=norm, scale_each=scale_each, padding=0, nrow=nrow, range=range)
        else:
            ten_grid = make_grid(ten, normalize=norm, scale_each=scale_each, padding=0, nrow=nrow)
        img_grid = ten_grid.data.cpu().numpy().transpose(1, 2, 0)
        img_grid = (img_grid * 255).astype(np.uint8)
        return img_grid

    def ten2tengrid(self, ten, norm=True, scale_each=True, padding=0, nrow=8):
        ten_grid = make_grid(ten, normalize=norm, scale_each=scale_each, padding=padding, nrow=nrow, range=(-1., 1.))
        return ten_grid

    def display_current_results(self, visuals, epoch, step, imgname=None, nrow=8, mode='train'):
        if self.tf_log:
            grid_list = []
            for ten in visuals:
                grid_list.append(self.ten2tengrid(ten[:nrow], nrow=nrow).cpu())
            grid = torch.cat(grid_list, 1)
            self.writer.add_image(os.path.join(mode, 'img'), grid, global_step=step)
            ####
            img_list = []
            for ten in visuals:
                img_list.append(self.ten2imggrid(ten[:nrow], nrow=nrow))
            img = np.concatenate(img_list, 0)
            import matplotlib.pyplot as plt
            plt.imsave(f"{self.img_dir}/{mode}_epoch{epoch}_step{step}.png", img)

        else:
            img_list = []
            for ten in visuals:
                img_list.append(self.ten2imggrid(ten[:nrow], nrow=nrow, range=(-1, 1)))
            img = np.concatenate(img_list, 0)
            import matplotlib.pyplot as plt
            if imgname is None:
                plt.imsave(f"{self.img_dir}/{mode}_epoch{epoch}_step{step}.png", img)
            else:
                imgname = imgname.split('/')[-1][:-4]
                plt.imsave(f"{self.img_dir}/{imgname}.png", img)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if 'input_label' == key or 'changed_label' == key:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    def save_images(self, webpage, visuals, image_path):
        visuals = self.convert_visuals_to_numpy(visuals)

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_current_errors(self, errors, step, mode='train'):
        if self.tf_log:
            for tag, value in errors.items():
                self.writer.add_scalar(os.path.join(mode, tag), value, global_step=step)
        # ipdb.set_trace()
        # print()
