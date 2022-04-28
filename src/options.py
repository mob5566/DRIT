import argparse

class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # model related
    self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    self.parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    self.parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    self.parser.add_argument('--zs_dim', type=int, default=8, help='the dimension the of the style space')
    self.parser.add_argument('--aux_masks', action='store_true', help='specified if auxiliary masks are provided')
    self.parser.add_argument('--aux_loss', type=str, default='ce', help='loss function of the segmentation decoder', choices=['ce', 'lovazs', 'focal', 'dice'])
    self.parser.add_argument('--aux_skip_conn', action='store_true', help='specified if auxiliary masks have skip connections')
    self.parser.add_argument('--aux_n_classes', type=int, default=2, help='number of classes for segmentation')
    self.parser.add_argument('--aux_cls_weights', nargs='*', type=float, default=None, help='weights of classes for loss function')

    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
    self.parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')
    self.parser.add_argument('--pair_align', action='store_true', help='specified if the dataset is paired')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='../logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='../results', help='path for saving result images and models')
    self.parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')
    self.parser.add_argument('--display_img_freq', type=int, default=5, help='freq (epoch) of display')
    self.parser.add_argument('--display_log_freq', type=int, default=10, help='freq (iteration) of display')
    self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

    # training related
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--gen_lr', type=float, default=0.0001, help='the initial learning rate for generator')
    self.parser.add_argument('--dis_lr', type=float, default=0.00005, help='the initial learning rate for discriminator')
    self.parser.add_argument('--dcont_lr', type=float, default=0.00004, help='the initial learning rate for discriminator of contents')
    self.parser.add_argument('--n_ep', type=int, default=500, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=0, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--weight_decay', type=float, default=0.0001, help='the weight decay of the optimizer')
    self.parser.add_argument('--max_it', type=int, default=500000, help='maximum number of iterations')
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')

  def parse(self, arguments=None):
    self.opt = self.parser.parse_args(arguments)
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt

class TestOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
    self.parser.add_argument('--crop_size', type=int, default=216, help='cropped image size for training')
    self.parser.add_argument('--nThreads', type=int, default=4, help='for data loader')
    self.parser.add_argument('--input_dim_a', type=int, default=3, help='# of input channels for domain A')
    self.parser.add_argument('--input_dim_b', type=int, default=3, help='# of input channels for domain B')
    self.parser.add_argument('--a2b', type=int, default=1, help='translation direction, 1 for a2b, 0 for b2a')
    self.parser.add_argument('--no_flip', action='store_true', help='specified if no flipping')
    self.parser.add_argument('--pair_align', action='store_true', help='specified if the dataset is paired')

    # ouptput related
    self.parser.add_argument('--num', type=int, default=5, help='number of outputs per image')
    self.parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='../outputs', help='path for saving result images and models')

    # model related
    self.parser.add_argument('--concat', type=int, default=1, help='concatenate attribute features for translation, set 0 for using feature-wise transform')
    self.parser.add_argument('--no_ms', action='store_true', help='disable mode seeking regularization')
    self.parser.add_argument('--resume', type=str, required=True, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
    self.parser.add_argument('--zs_dim', type=int, default=8, help='the dimension the of the style space')
    self.parser.add_argument('--aux_masks', action='store_true', help='specified if auxiliary masks are provided')
    self.parser.add_argument('--aux_skip_conn', action='store_true', help='specified if auxiliary masks have skip connections')
    self.parser.add_argument('--aux_n_classes', type=int, default=2, help='number of classes for segmentation')
    self.parser.add_argument('--aux_cls_weights', nargs='*', type=float, default=None, help='weights of classes for loss function')
    self.parser.add_argument('--lr', type=float, default=0.0001, help='the initial learning rate')
    self.parser.add_argument('--weight_decay', type=float, default=0.0001, help='the weight decay of the optimizer')

  def parse(self, arguments=None):
    self.opt = self.parser.parse_args(arguments)
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    # set irrelevant options
    self.opt.dis_scale = 3
    self.opt.dis_norm = 'None'
    self.opt.dis_spectral_norm = False
    return self.opt
