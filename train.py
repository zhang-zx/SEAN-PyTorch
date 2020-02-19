import sys

import data
from options.train_options import TrainOptions
from trainers.pix2pix_sean_trainer import Pix2PixSEANTrainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixSEANTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            loss_dict = trainer.get_latest_losses()
            errors = {k: v.mean().cpu().data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            visualizer.print_current_errors(epoch, iter_counter.total_steps_so_far, errors, iter_counter.time_per_iter)
            visualizer.plot_current_errors(errors, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = [
                (data_i['label'] / (opt.label_nc - 1) - 0.5) / 0.5, trainer.get_latest_generated(), data_i['image']
            ]
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
            epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
