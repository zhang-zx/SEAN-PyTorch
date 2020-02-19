import os
from collections import OrderedDict

import data
from models.pix2pix_sean_model import Pix2PixSEANModel
from options.test_options import TestOptions
from util import html
from util.visualizer import Visualizer

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixSEANModel(opt)
model.eval()

visualizer = Visualizer(opt, 'test')

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    ref_image, reconst, ref_reconst = model(data_i, mode='region_transfer_test', region_index=opt.region_index)
    # import ipdb
    # ipdb.set_trace()
    changed_label = data_i['label'].clone()
    changed_label[changed_label != opt.region_index] = 0
    img_path = data_i['path']
    for b in range(ref_image.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('source_image', data_i['image'][b]),
                               ('input_label', data_i['label'][b]),
                               ('reconst', reconst[b]),
                               ('ref_image', ref_image[b]),
                               ('changed_label', changed_label[b]),
                               ('ref_reconst', ref_reconst[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])
    visuals = [
        data_i['image'], (data_i['label'].float() / (opt.label_nc - 1.) - 0.5) / 0.5, reconst, ref_image,
                         (changed_label.float() / (opt.label_nc - 1) - 0.5) / 0.5, ref_reconst
    ]
    visualizer.display_current_results(visuals, opt.which_epoch, i, mode='test')

webpage.save()
