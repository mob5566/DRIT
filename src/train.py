import torch
from options import TrainOptions
from dataset import dataset_unpair
from model import DRIT
from saver import Saver
from tqdm import tqdm
import signal


def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()

  # handle early stopping
  early_stop = False
  def signal_handler(sig, frame):
    nonlocal early_stop
    if not early_stop:
      early_stop = True

  signal.signal(signal.SIGINT, signal_handler)

  # daita loader
  print('\n--- load dataset ---')
  dataset = dataset_unpair(opts)
  N = len(dataset)
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(opts.gpu)
  if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_it = 0
  else:
    ep0, total_it = model.resume(opts.resume)
  model.set_scheduler(opts, last_ep=ep0)
  ep0 += 1
  print('start the training at epoch %d'%(ep0))

  # saver for display and output
  saver = Saver(opts)

  # train
  print('\n--- train ---')
  max_it = opts.max_it
  for ep in tqdm(range(ep0, opts.n_ep), unit='epoch'):
    lr = model.gen_opt.param_groups[0]["lr"]

    for it, data in tqdm(enumerate(train_loader), total=N,
                         desc=f'Epoch {ep} (lr: {lr:08f})'):
      if opts.aux_masks:
        images_a, masks_a, images_b = data
      else:
        images_a, images_b = data
        masks_a = None

      if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
        continue

      # input data
      images_a = images_a.cuda(opts.gpu).detach()
      images_b = images_b.cuda(opts.gpu).detach()
      if opts.aux_masks:
        masks_a = masks_a.cuda(opts.gpu).detach()

      # update model
      if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
        model.update_D_content(images_a, images_b)
        continue
      else:
        model.update_D(images_a, images_b, masks_a)
        model.update_EG()

      # save to display file
      if not opts.no_display_img:
        saver.write_display_log(total_it, model)

      total_it += 1
      if total_it >= max_it or early_stop:
        saver.write_img(ep, model, True)
        saver.write_model(ep, total_it, model, True)
        break

    # decay learning rate
    if opts.n_ep_decay >= 0 and ep >= opts.n_ep_decay:
      model.update_lr()

    # display images to tensorboard
    saver.write_display_img(ep, model)

    # save result image
    saver.write_img(ep, model)

    # Save network weights
    saver.write_model(ep, total_it, model)

    if total_it >= max_it or early_stop:
      break

  return

if __name__ == '__main__':
  main()
