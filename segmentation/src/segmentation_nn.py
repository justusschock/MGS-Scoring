from shutil import copyfile

from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from network import Network
from image_handler import ImageHandler
import os

import time

from unaligned_data_loader import UnalignedDataLoader
from correspondence_data_loader import CorrespondenceDataLoader


class SegmentationNetwork(object):

    def __init__(self, options, gpu_ids=[]):
        self.options = options
        self.model = Network(n_input_channels=options.input_nc,
                             n_output_channels=options.output_nc,
                             threshold_gen=options.threshold_gen,
                             n_blocks=options.n_blocks,
                             initial_filters=options.initial_filters,
                             dropout_value=options.dropout_value,
                             kernel_size=options.kernel_size,
                             strides=options.strides,
                             lr=options.lr,
                             decay=options.decay,
                             decay_epochs=options.decay_epochs,
                             batch_size=options.batch_size,
                             image_width=options.image_width,
                             image_height=options.image_height,
                             pool_size=options.pool_size,
                             lambda_gan=0,
                             load_network=options.load_network,
                             gen_label=options.gen_label,
                             dis_label=options.dis_label,
                             load_epoch=options.load_epoch,
                             model_path=os.path.join(options.model_path, options.name),
                             name=options.name,
                             gpu_ids=gpu_ids)


        self.data_set = CorrespondenceDataLoader(options).load_data()
        self.image_handler = ImageHandler()
        self.loss_dir = self.options.output_path + "/" + self.options.name + "/Train"
        copyfile(os.path.relpath('seg_config.yaml'), os.path.join(self.options.model_path, self.options.name, 'seg_config.yaml'))

    def train(self):
        """
        Function to train model from dataset
        :return: None
        """
        print("Started Training")
        batch_size = self.options.batch_size
        loss_file = self.loss_dir + "/losses.txt"

        if os.path.isfile(loss_file):
            if self.options.load_network:
                self.erase_loss_file(loss_file, self.options.load_epoch)
            else:
                self.erase_loss_file(loss_file, 0)

        if self.options.load_network:
            base_epoch = int(self.options.load_epoch)
        else:
            base_epoch = 0

        for epoch in range(1 + base_epoch, self.options.n_epochs + self.options.decay_epochs + 1 + base_epoch):

            if epoch >= self.options.epoch_start_discriminiator:
                self.model.lambda_gan = self.options.lambda_gan

            epoch_start_time = time.time()

            steps = 0
            t = 0

            for i, data in enumerate(self.data_set):
                iter_start_time = time.time()

                current_batch = data
                self.model.set_input(current_batch['img'], current_batch['mask'])

                self.model.optimize_g()

                if (steps + 1) % self.options.print_freq == 0:
                    if self.options.train_D:
                        errors = self.model.get_current_errors_d()
                    else:
                        errors = self.model.get_current_errors()
                    t = (time.time() - iter_start_time)

                    message = '(epoch: %d, step: %d, time/step: %.3f) ' % (epoch, steps + 1, t)
                    for k, v in errors.items():
                        message += '%s: %.3f, ' % (k, v)

                    if not os.path.isdir(str(self.loss_dir)):
                        os.makedirs(str(self.loss_dir))

                    with open(loss_file, 'a+') as f:
                        f.write(message + "\n")

                    print(message)

                steps += 1
                if steps >= self.options.steps_per_epoch:
                    break

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, self.options.n_epochs + self.options.decay_epochs, time.time() - epoch_start_time))

            if epoch % self.options.save_img_freq == 0:
                # self.model.predict()
                output_dir = self.options.output_path + "/" + self.options.name + "/Train/images"
                self.image_handler.save_image(self.model.var_img, output_dir, 'epoch_%03d_real_img' % epoch)
                if not self.options.train_D:
                    self.image_handler.save_mask(self.model.fake_mask, output_dir, 'epoch_%03d_fake_mask' % epoch)
                self.image_handler.save_mask(self.model.var_gt, output_dir, 'epoch_%03d_gt' % epoch)

                self.create_html_file(epoch)

            if epoch % self.options.save_freq == 0:
                print('saving the model at the end of epoch %d' % epoch)
                self.model.save(str(epoch))

            if not self.options.train_D:
                self.plot_losses(loss_file)

            if epoch > self.options.n_epochs:
                self.model.update_learning_rate()

    def predict(self, n_predicts=0):
        """
        Function to predict Images from Dataroot to a subfolder of output_path
        :param n_predicts: number of Images to predict, set 0 to predict all images
        :return:
        """

        print("Started Prediction")
        if not n_predicts:
            n_predicts = len(self.data_set)

        for i, data in enumerate(self.data_set):
            self.model.set_input(data['img'])
            predicted_mask = self.model.predict()

            # FIXME: data['path_img'] gives list of strings instead of string
            save_name = (os.path.split(data['path_img'][0])[-1]).rsplit('.', 1)[0] + '_pred'

            output_dir = self.options.output_path + "/" + self.options.name + "/Predict/" + "Epoch_" + str(self.options.load_epoch)
            self.image_handler.save_mask(predicted_mask, output_dir, save_name)

            if ((i + 1) % 10) == 0:
                print("Predicted %d of %d Images" % (i+1, n_predicts))

            if (i + 1) >= n_predicts:
                break

        print("Finished Prediction")



    @staticmethod
    def erase_loss_file(loss_file, initial_epoch):
        """
        Function to erase all losses of future epochs
        Necessary for continued train with intermediate epoch or restart training with same name
        :param loss_file: file the losses are stored in
        :param initial_epoch: epoch to start training
        :return: None
        """

        new_content = []

        with open(loss_file, 'r') as f:
            content = f.readlines()

            for line in content:
                header, loss_data = line.split(")", maxsplit=1)

                header_value_paires = header.split(",")
                epoch = int(header_value_paires[0].split(":")[1])
                if epoch < initial_epoch:
                    new_content.append(line)

        with open(loss_file, 'w') as f:
            for line in new_content:
                f.write(line)


    def plot_losses(self, loss_file):
        """
        Function to plot loss values
        :param loss_file: file to read loss values from
        :return: None
        """
        if not os.path.isfile(loss_file):
            raise ValueError('%s is not a file' % str(loss_file))

        g_seg_losses = []
        epochs = []
        steps = []

        with open(loss_file, 'r') as f:
            content = f.readlines()

        content = [x.strip(" ").strip("\n") for x in content]

        for line in content:
            header, loss_data = line.split(")", maxsplit=1)

            header_value_paires = header.split(",")
            epoch = int(header_value_paires[0].split(":")[1])
            step = int(header_value_paires[1].split(":")[1])

            step_total = (epoch-1)*self.options.steps_per_epoch + step

            _tmp = str(loss_data).split(",")
            g_seg = _tmp[0]

            g_seg_losses.append(float(g_seg.split(":")[1]))
            epochs.append(epoch)
            steps.append(step_total)

        markers = {0: "o",
                   1: "s",
                   2: "^",
                   3: "D",
                   4: "*",
                   5: "x"
                   }

        colors = {0: "b",
                  1: "g",
                  2: "r",
                  3: "c",
                  4: "m",
                  5: "k",
                  6: "y"
                  }

        print("plotting Errors and save files to ", self.loss_dir)
        fig_losses_steps = plt.figure(1, figsize=(48, 27))
        fig_losses_epochs = plt.figure(2, figsize=(48, 27))

        figures = [fig_losses_steps, fig_losses_epochs]
        loss_labels = ["Loss Seg"]

        loss_list = [g_seg_losses]

        time_list = [steps]

        time_labels = ["Total Steps",
                       "Epochs"]

        save_paths = [self.loss_dir + "/loss_plot_steps.png",
                      self.loss_dir + "/loss_plot_epochs.png"]

        max_epoch = max(epochs)

        for j in range(len(time_list)):
            plt.figure(j + 1)
            for i, loss in enumerate(loss_list):
                ax = figures[j].add_subplot(len(loss_list), 1, i+1)
                style = markers[i % 6] + colors[i % 7] + "-"
                ax.plot(time_list[j], loss_list[i], style, label=loss_labels[i], markersize=3)
                ax.set_title(loss_labels[i])
                ax.set_xlabel(time_labels[j])
                ax.set_ylabel("Loss Values")
                if j == 0:
                    for ep in range(1, max_epoch + 1):
                        ax.axvline(ep * self.options.steps_per_epoch)

            figures[j].subplots_adjust(hspace=1.0)
            figures[j].savefig(save_paths[j])

    

    def create_html_file(self, current_epoch, width=400):
        """
        Function to create HTML file for better visualization
        :param current_epoch: current epoch (epoch shown at top of the HTML file)
        :param width: width of displayed images
        :return: None
        """
        print("Create HTML File")
        epoch_freq = self.options.save_img_freq

        web_dir = self.options.output_path + "/" + self.options.name + "/Train"
        self.image_handler.create_html_file(web_dir, "OverviewEpochs", "./images",
                                            current_epoch, epoch_freq, width)
