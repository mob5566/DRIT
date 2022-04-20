import os
import torch.utils.data as data
import numpy as np

from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
from torch import get_rng_state, set_rng_state, from_numpy
import random

class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.size = len(self.img)
    self.input_dim = input_dim

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size

class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot
    self.aux_masks = opts.aux_masks

    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]
    if self.aux_masks:
        self.Amask = [os.path.join(self.dataroot, opts.phase + 'A_ann', x) for x in images_A]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      if self.aux_masks:
        data_A, mask_A = self.load_img(self.A[index], self.input_dim_A,
                                       msk_name=self.Amask[index])
      else:
        data_A = self.load_img(self.A[index], self.input_dim_A)

      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      if self.aux_masks:
        rnd_index = random.randint(0, self.A_size - 1)
        data_A, mask_A = self.load_img(self.A[rnd_index], self.input_dim_A,
                                       msk_name=self.Amask[rnd_index])
      else:
        data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)

      data_B = self.load_img(self.B[index], self.input_dim_B)

    return (data_A, mask_A, data_B) if self.aux_masks else (data_A, data_B)

  def load_img(self, img_name, input_dim, msk_name=None):
    img = Image.open(img_name).convert('RGB')

    if msk_name is not None:
        msk = Image.open(msk_name).convert('P')
        state = get_rng_state()
        img = self.transforms(img)
        set_rng_state(state)
        msk = self.transforms(msk)
        msk = from_numpy(np.array(msk, dtype=np.uint8)).long()
    else:
        img = self.transforms(img)

    img = ToTensor()(img)
    img = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)

    return (img, msk) if msk_name is not None else img

  def __len__(self):
    return self.dataset_size
