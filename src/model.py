import networks
import torch
import torch.nn as nn

from mmseg.models.losses import LovaszLoss, DiceLoss, FocalLoss

def _plot_mask(mask):
  colormap = torch.Tensor([
    [0xff, 0xff, 0xff], [0x56, 0xb8, 0x81], [0xf1, 0xf0, 0x75],
    [0x8a, 0x8a, 0xcb], [0xfb, 0xb0, 0x3b], [0x3b, 0xb2, 0xd0],
    [0xed, 0x64, 0x98], [0x40, 0x40, 0x40]
  ]).cpu()
  newmap = colormap[torch.argmax(mask, dim=1)].permute(0, 3, 1, 2)
  newmap = (newmap / 255.0) * 2.0 - 1.0

  return newmap

class DRIT(nn.Module):
  def __init__(self, opts):
    super(DRIT, self).__init__()

    # parameters
    gen_lr = opts.gen_lr
    dis_lr = opts.dis_lr
    lr_dcontent = opts.dcont_lr
    self.nz = opts.zs_dim
    self.concat = opts.concat
    self.no_ms = opts.no_ms
    self.aux_masks = opts.aux_masks
    self.bsize = opts.batch_size
    self.weight_decay = opts.weight_decay
    self.aux_weights = None
    # self.opts = opts

    if self.aux_masks and opts.aux_cls_weights is not None:
        assert len(opts.aux_cls_weights) == opts.aux_n_classes, f'The length of auxiliary weights ({len(opts.aux_cls_weights)}) should be equal to the number of classes ({opts.aux_n_classes})'
        self.aux_weights = torch.Tensor(opts.aux_cls_weights)

    # discriminators
    if opts.dis_scale > 1:
      self.disA = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disA2 = networks.MultiScaleDis(opts.input_dim_a, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB2 = networks.MultiScaleDis(opts.input_dim_b, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    else:
      self.disA = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disA2 = networks.Dis(opts.input_dim_a, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
      self.disB2 = networks.Dis(opts.input_dim_b, norm=opts.dis_norm, sn=opts.dis_spectral_norm)
    self.disContent = networks.Dis_content()

    # encoders
    self.enc_c = networks.E_content(opts.input_dim_a, opts.input_dim_b)
    self.enc_c_shared = networks.E_content_shared()
    if self.concat:
      self.enc_a = networks.E_attr_concat(
          opts.input_dim_a, opts.input_dim_b, self.nz, norm_layer=None,
          nl_layer=networks.get_non_linearity(layer_type='lrelu')
      )
    else:
      self.enc_a = networks.E_attr(opts.input_dim_a, opts.input_dim_b, self.nz)

    # generator
    if self.concat:
      self.gen = networks.G_concat(opts.input_dim_a, opts.input_dim_b, nz=self.nz)
    else:
      self.gen = networks.G(opts.input_dim_a, opts.input_dim_b, nz=self.nz)

    # segmentation decoders
    if self.aux_masks:
      self.unet_dec = networks.UNetDecoder(n_classes=opts.aux_n_classes,
                                           skip_conn=opts.aux_skip_conn)

    # optimizers
    self.disA_opt = torch.optim.Adam(self.disA.parameters(), lr=dis_lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
    self.disB_opt = torch.optim.Adam(self.disB.parameters(), lr=dis_lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
    self.disA2_opt = torch.optim.Adam(self.disA2.parameters(), lr=dis_lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
    self.disB2_opt = torch.optim.Adam(self.disB2.parameters(), lr=dis_lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
    self.disContent_opt = torch.optim.Adam(self.disContent.parameters(), lr=lr_dcontent, betas=(0.5, 0.999), weight_decay=self.weight_decay)
    self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=gen_lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
    self.enc_c_shared_opt = torch.optim.Adam(self.enc_c_shared.parameters(), lr=gen_lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
    self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=gen_lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=gen_lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

    if self.aux_masks:
      self.unet_dec_opt = torch.optim.Adam(self.unet_dec.parameters(), lr=gen_lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

    # Setup the loss function for training
    self.criterionL1 = torch.nn.L1Loss()
    if self.aux_masks:
      if opts.aux_loss == 'lovazs':
        self.criterionSeg = LovaszLoss(per_image=True, class_weight=self.aux_weights)
      elif opts.aux_loss == 'focal':
        self.criterionSeg = FocalLoss(class_weight=self.aux_weights)
      elif opts.aux_loss == 'dice':
        self.criterionSeg = DiceLoss(class_weight=self.aux_weights)
      else:
        self.criterionSeg = torch.nn.CrossEntropyLoss(weight=self.aux_weights)

  def initialize(self):
    self.disA.apply(networks.gaussian_weights_init)
    self.disB.apply(networks.gaussian_weights_init)
    self.disA2.apply(networks.gaussian_weights_init)
    self.disB2.apply(networks.gaussian_weights_init)
    self.disContent.apply(networks.gaussian_weights_init)
    self.gen.apply(networks.gaussian_weights_init)
    if self.aux_masks:
      self.unet_dec.apply(networks.gaussian_weights_init)
    self.enc_c.apply(networks.gaussian_weights_init)
    self.enc_c_shared.apply(networks.gaussian_weights_init)
    self.enc_a.apply(networks.gaussian_weights_init)

  def set_scheduler(self, opts, last_ep=0):
    self.disA_sch = networks.get_scheduler(self.disA_opt, opts, last_ep)
    self.disB_sch = networks.get_scheduler(self.disB_opt, opts, last_ep)
    self.disA2_sch = networks.get_scheduler(self.disA2_opt, opts, last_ep)
    self.disB2_sch = networks.get_scheduler(self.disB2_opt, opts, last_ep)
    self.disContent_sch = networks.get_scheduler(self.disContent_opt, opts, last_ep)
    self.enc_c_sch = networks.get_scheduler(self.enc_c_opt, opts, last_ep)
    self.enc_c_shared_sch = networks.get_scheduler(self.enc_c_shared_opt, opts, last_ep)
    self.enc_a_sch = networks.get_scheduler(self.enc_a_opt, opts, last_ep)
    self.gen_sch = networks.get_scheduler(self.gen_opt, opts, last_ep)
    if self.aux_masks:
      self.unet_dec_sch = networks.get_scheduler(self.unet_dec_opt, opts, last_ep)

  def setgpu(self, gpu):
    self.gpu = gpu
    self.disA.cuda(self.gpu)
    self.disB.cuda(self.gpu)
    self.disA2.cuda(self.gpu)
    self.disB2.cuda(self.gpu)
    self.disContent.cuda(self.gpu)
    self.enc_c.cuda(self.gpu)
    self.enc_c_shared.cuda(self.gpu)
    self.enc_a.cuda(self.gpu)
    self.gen.cuda(self.gpu)
    if self.aux_masks:
      self.unet_dec.cuda(self.gpu)
      self.criterionSeg.cuda(self.gpu)

  def get_z_random(self, batchSize, nz, random_type='gauss'):
    z = torch.randn(batchSize, nz).cuda(self.gpu)
    return z

  def test_forward(self, image, a2b=True):
    self.z_random = self.get_z_random(image.size(0), self.nz, 'gauss')
    if a2b:
        (self.z_content, self.z_content_outs) = self.enc_c.forward_a(image)
        self.z_content = self.enc_c_shared.forward_a(self.z_content)
        output = self.gen.forward_b(self.z_content, self.z_random)
    else:
        (self.z_content, self.z_content_outs) = self.enc_c.forward_b(image)
        self.z_content = self.enc_c_shared.forward_b(self.z_content)
        output = self.gen.forward_a(self.z_content, self.z_random)
    return output

  def test_forward_transfer(self, image_a, image_b, a2b=True):
    (self.z_content_a, self.z_content_a_outs), (self.z_content_b, self.z_content_b_outs) = self.enc_c(image_a, image_b)
    self.z_content_a, self.z_content_b = self.enc_c_shared(self.z_content_a, self.z_content_b)
    if self.concat:
      self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a(image_a, image_b)
      std_a = self.logvar_a.mul(0.5).exp_()
      eps = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_a = eps.mul(std_a).add_(self.mu_a)
      std_b = self.logvar_b.mul(0.5).exp_()
      eps = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_b = eps.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_a, self.z_attr_b = self.enc_a(image_a, image_b)
    if a2b:
      output = self.gen.forward_b(self.z_content_a, self.z_attr_b)
    else:
      output = self.gen.forward_a(self.z_content_b, self.z_attr_a)
    return output

  def forward(self):
    # input images
    half_size = self.bsize // 2
    real_A = self.input_A
    real_B = self.input_B
    self.real_A_encoded = real_A[0:half_size]
    self.real_A_random = real_A[half_size:]
    self.real_B_encoded = real_B[0:half_size]
    self.real_B_random = real_B[half_size:]
    if self.aux_masks:
      mask_A = self.mask_A
      self.A_mask = mask_A[:half_size]

    # get encoded z_c
    (self.z_content_a, self.z_content_a_outs), (self.z_content_b, self.z_content_b_outs) = self.enc_c(self.real_A_encoded, self.real_B_encoded)
    self.z_content_a, self.z_content_b = self.enc_c_shared(self.z_content_a, self.z_content_b)

    # get segmentation
    if self.aux_masks:
      self.real_A_mask = self.unet_dec(self.z_content_a, self.z_content_a_outs[:2][::-1])
      self.real_B_mask = self.unet_dec(self.z_content_b, self.z_content_b_outs[:2][::-1])

    # get encoded z_a
    if self.concat:
      self.mu_a, self.logvar_a, self.mu_b, self.logvar_b = self.enc_a(self.real_A_encoded, self.real_B_encoded)
      std_a = self.logvar_a.mul(0.5).exp_()
      eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_a = eps_a.mul(std_a).add_(self.mu_a)
      std_b = self.logvar_b.mul(0.5).exp_()
      eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_b = eps_b.mul(std_b).add_(self.mu_b)
    else:
      self.z_attr_a, self.z_attr_b = self.enc_a(self.real_A_encoded, self.real_B_encoded)

    # get random z_a
    self.z_random = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')
    if not self.no_ms:
      self.z_random2 = self.get_z_random(self.real_A_encoded.size(0), self.nz, 'gauss')

    # first cross translation
    if not self.no_ms:
      input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b, self.z_content_b),0)
      input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a, self.z_content_a),0)
      input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random, self.z_random2),0)
      input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random, self.z_random2),0)
      output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
      output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
      self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random, self.fake_A_random2 = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
      self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random, self.fake_B_random2 = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)
    else:
      input_content_forA = torch.cat((self.z_content_b, self.z_content_a, self.z_content_b),0)
      input_content_forB = torch.cat((self.z_content_a, self.z_content_b, self.z_content_a),0)
      input_attr_forA = torch.cat((self.z_attr_a, self.z_attr_a, self.z_random),0)
      input_attr_forB = torch.cat((self.z_attr_b, self.z_attr_b, self.z_random),0)
      output_fakeA = self.gen.forward_a(input_content_forA, input_attr_forA)
      output_fakeB = self.gen.forward_b(input_content_forB, input_attr_forB)
      self.fake_A_encoded, self.fake_AA_encoded, self.fake_A_random = torch.split(output_fakeA, self.z_content_a.size(0), dim=0)
      self.fake_B_encoded, self.fake_BB_encoded, self.fake_B_random = torch.split(output_fakeB, self.z_content_a.size(0), dim=0)

    # get reconstructed encoded z_c
    (self.z_content_recon_b, self.z_content_recon_b_outs), (self.z_content_recon_a, self.z_content_recon_a_outs) = self.enc_c(self.fake_A_encoded, self.fake_B_encoded)
    self.z_content_recon_b, self.z_content_recon_a = self.enc_c_shared(self.z_content_recon_b, self.z_content_recon_a)

    # get segmentation
    if self.aux_masks:
      self.recon_A_mask = self.unet_dec(self.z_content_recon_a, self.z_content_recon_a_outs[:2][::-1])
      self.recon_B_mask = self.unet_dec(self.z_content_recon_b, self.z_content_recon_b_outs[:2][::-1])

    # get reconstructed encoded z_a
    if self.concat:
      self.mu_recon_a, self.logvar_recon_a, self.mu_recon_b, self.logvar_recon_b = self.enc_a(self.fake_A_encoded, self.fake_B_encoded)
      std_a = self.logvar_recon_a.mul(0.5).exp_()
      eps_a = self.get_z_random(std_a.size(0), std_a.size(1), 'gauss')
      self.z_attr_recon_a = eps_a.mul(std_a).add_(self.mu_recon_a)
      std_b = self.logvar_recon_b.mul(0.5).exp_()
      eps_b = self.get_z_random(std_b.size(0), std_b.size(1), 'gauss')
      self.z_attr_recon_b = eps_b.mul(std_b).add_(self.mu_recon_b)
    else:
      self.z_attr_recon_a, self.z_attr_recon_b = self.enc_a(self.fake_A_encoded, self.fake_B_encoded)

    # second cross translation
    self.fake_A_recon = self.gen.forward_a(self.z_content_recon_a, self.z_attr_recon_a)
    self.fake_B_recon = self.gen.forward_b(self.z_content_recon_b, self.z_attr_recon_b)

    # for display
    if not self.aux_masks:
      self.image_display = torch.cat((self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(),
                                      self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(), self.fake_A_recon[0:1].detach().cpu(),
                                      self.real_B_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(),
                                      self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)
    else:
      self.A_mask_display = _plot_mask(self.real_A_mask[:1].detach().cpu())
      self.B_mask_display = _plot_mask(self.real_B_mask[:1].detach().cpu())
      self.image_display = torch.cat((self.A_mask_display.detach().cpu(), self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(),
                                      self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(), self.fake_A_recon[0:1].detach().cpu(),
                                      self.B_mask_display.detach().cpu(), self.real_B_encoded[0:1].detach().cpu(), self.fake_A_encoded[0:1].detach().cpu(),
                                      self.fake_A_random[0:1].detach().cpu(), self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)

    # for latent regression
    if self.concat:
      self.mu2_a, _, self.mu2_b, _ = self.enc_a(self.fake_A_random, self.fake_B_random)
    else:
      self.z_attr_random_a, self.z_attr_random_b = self.enc_a(self.fake_A_random, self.fake_B_random)

  def forward_content(self):
    half_size = self.bsize // 2
    self.real_A_encoded = self.input_A[0:half_size]
    self.real_B_encoded = self.input_B[0:half_size]
    # get encoded z_c
    (self.z_content_a, self.z_content_a_outs), (self.z_content_b, self.z_content_b_outs) = self.enc_c(self.real_A_encoded, self.real_B_encoded)
    self.z_content_a, self.z_content_b = self.enc_c_shared(self.z_content_a, self.z_content_b)

  def update_D_content(self, image_a, image_b):
    self.input_A = image_a
    self.input_B = image_b
    self.forward_content()
    self.disContent_opt.zero_grad()
    loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
    self.disContent_loss = loss_D_Content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def update_D(self, image_a, image_b, mask_a=None):
    self.input_A = image_a
    self.input_B = image_b
    if self.aux_masks:
      self.mask_A = mask_a
    self.forward()

    # update disA
    self.disA_opt.zero_grad()
    loss_D1_A = self.backward_D(self.disA, self.real_A_encoded, self.fake_A_encoded)
    self.disA_loss = loss_D1_A.item()
    self.disA_opt.step()

    # update disA2
    self.disA2_opt.zero_grad()
    loss_D2_A = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random)
    self.disA2_loss = loss_D2_A.item()
    if not self.no_ms:
      loss_D2_A2 = self.backward_D(self.disA2, self.real_A_random, self.fake_A_random2)
      self.disA2_loss += loss_D2_A2.item()
    self.disA2_opt.step()

    # update disB
    self.disB_opt.zero_grad()
    loss_D1_B = self.backward_D(self.disB, self.real_B_encoded, self.fake_B_encoded)
    self.disB_loss = loss_D1_B.item()
    self.disB_opt.step()

    # update disB2
    self.disB2_opt.zero_grad()
    loss_D2_B = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random)
    self.disB2_loss = loss_D2_B.item()
    if not self.no_ms:
      loss_D2_B2 = self.backward_D(self.disB2, self.real_B_random, self.fake_B_random2)
      self.disB2_loss += loss_D2_B2.item()
    self.disB2_opt.step()

    # update disContent
    self.disContent_opt.zero_grad()
    loss_D_Content = self.backward_contentD(self.z_content_a, self.z_content_b)
    self.disContent_loss = loss_D_Content.item()
    nn.utils.clip_grad_norm_(self.disContent.parameters(), 5)
    self.disContent_opt.step()

  def backward_D(self, netD, real, fake):
    pred_fake = netD(fake.detach())
    pred_real = netD(real.detach())
    loss_D = 0
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all0 = torch.zeros_like(out_fake).cuda(self.gpu)
      all1 = torch.ones_like(out_real).cuda(self.gpu)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      loss_D += ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def backward_contentD(self, imageA, imageB):
    pred_fake = self.disContent(imageA.detach())
    pred_real = self.disContent(imageB.detach())
    for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
      out_fake = nn.functional.sigmoid(out_a)
      out_real = nn.functional.sigmoid(out_b)
      all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
      all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
      ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
      ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
    loss_D = ad_true_loss + ad_fake_loss
    loss_D.backward()
    return loss_D

  def update_EG(self):
    # update G, Ec, Ea
    self.enc_c_opt.zero_grad()
    self.enc_c_shared_opt.zero_grad()
    self.enc_a_opt.zero_grad()
    self.gen_opt.zero_grad()
    self.backward_EG()
    self.backward_G_alone()   # backward_G_alone here to accumulate gradient together before update
    self.enc_c_opt.step()
    self.enc_c_shared_opt.step()
    self.enc_a_opt.step()
    self.gen_opt.step()

    # update G, Ec
    # self.enc_c_opt.zero_grad()
    # self.gen_opt.zero_grad()
    # # self.forward()                   # must forward second time before update for new pytorch version
    # self.backward_G_alone()
    # self.enc_c_opt.step()
    # self.gen_opt.step()

  def backward_EG(self):
    # content Ladv for generator
    loss_G_GAN_Acontent = self.backward_G_GAN_content(self.z_content_a)
    loss_G_GAN_Bcontent = self.backward_G_GAN_content(self.z_content_b)

    # Ladv for generator
    loss_G_GAN_A = self.backward_G_GAN(self.fake_A_encoded, self.disA)
    loss_G_GAN_B = self.backward_G_GAN(self.fake_B_encoded, self.disB)

    # KL loss - z_a
    if self.concat:
      kl_element_a = self.mu_a.pow(2).add_(self.logvar_a.exp()).mul_(-1).add_(1).add_(self.logvar_a)
      loss_kl_za_a = torch.sum(kl_element_a).mul_(-0.5) * 0.01
      kl_element_b = self.mu_b.pow(2).add_(self.logvar_b.exp()).mul_(-1).add_(1).add_(self.logvar_b)
      loss_kl_za_b = torch.sum(kl_element_b).mul_(-0.5) * 0.01
    else:
      loss_kl_za_a = self._l2_regularize(self.z_attr_a) * 0.01
      loss_kl_za_b = self._l2_regularize(self.z_attr_b) * 0.01

    # KL loss - z_c
    loss_kl_zc_a = self._l2_regularize(self.z_content_a) * 0.01
    loss_kl_zc_b = self._l2_regularize(self.z_content_b) * 0.01

    # cross cycle consistency loss
    loss_G_L1_A = self.criterionL1(self.fake_A_recon, self.real_A_encoded) * 10
    loss_G_L1_B = self.criterionL1(self.fake_B_recon, self.real_B_encoded) * 10
    loss_G_L1_AA = self.criterionL1(self.fake_AA_encoded, self.real_A_encoded) * 10
    loss_G_L1_BB = self.criterionL1(self.fake_BB_encoded, self.real_B_encoded) * 10

    # segmentation loss
    if self.aux_masks:
      loss_seg_a = self.criterionSeg(self.real_A_mask, self.A_mask) * 10
      loss_seg_a_recon = self.criterionSeg(self.recon_A_mask, self.A_mask) * 10

    loss_G = loss_G_GAN_A + loss_G_GAN_B + \
             loss_G_GAN_Acontent + loss_G_GAN_Bcontent + \
             loss_G_L1_AA + loss_G_L1_BB + \
             loss_G_L1_A + loss_G_L1_B + \
             loss_kl_zc_a + loss_kl_zc_b + \
             loss_kl_za_a + loss_kl_za_b

    if self.aux_masks:
      loss_G += loss_seg_a + loss_seg_a_recon

    loss_G.backward(retain_graph=True)

    self.gan_loss_a = loss_G_GAN_A.item()
    self.gan_loss_b = loss_G_GAN_B.item()
    self.gan_loss_acontent = loss_G_GAN_Acontent.item()
    self.gan_loss_bcontent = loss_G_GAN_Bcontent.item()
    self.kl_loss_za_a = loss_kl_za_a.item()
    self.kl_loss_za_b = loss_kl_za_b.item()
    self.kl_loss_zc_a = loss_kl_zc_a.item()
    self.kl_loss_zc_b = loss_kl_zc_b.item()
    self.l1_recon_A_loss = loss_G_L1_A.item()
    self.l1_recon_B_loss = loss_G_L1_B.item()
    self.l1_recon_AA_loss = loss_G_L1_AA.item()
    self.l1_recon_BB_loss = loss_G_L1_BB.item()
    if self.aux_masks:
      self.seg_a_loss = loss_seg_a.item()
      self.seg_a_recon_loss = loss_seg_a_recon.item()
    self.G_loss = loss_G.item()

  def backward_G_GAN_content(self, data):
    outs = self.disContent(data)
    # model = networks.Dis_content()
    # model.load_state_dict(self.disContent.state_dict())
    # model.cuda(self.gpu)
    # outs = model.forward(data)

    ad_loss = 0
    for out in outs:
      outputs_fake = nn.functional.sigmoid(out)
      all_half = 0.5*torch.ones((outputs_fake.size(0))).cuda(self.gpu)
      ad_loss += nn.functional.binary_cross_entropy(outputs_fake, all_half)
    return ad_loss

  def backward_G_GAN(self, fake, netD=None):
    outs_fake = netD(fake)
    # if self.opts.dis_scale > 1:
    #   model = networks.MultiScaleDis(self.opts.input_dim_a, self.opts.dis_scale, norm=self.opts.dis_norm, sn=self.opts.dis_spectral_norm)
    # else:
    #   model = networks.Dis(self.opts.input_dim_a, norm=self.opts.dis_norm, sn=self.opts.dis_spectral_norm)
    # model.load_state_dict(netD.state_dict())
    # model.cuda(self.gpu)
    # outs_fake = model.forward(fake)

    loss_G = 0
    for out_a in outs_fake:
      outputs_fake = nn.functional.sigmoid(out_a)
      all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
      loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
    return loss_G

  def backward_G_alone(self):
    # Ladv for generator
    loss_G_GAN2_A = self.backward_G_GAN(self.fake_A_random, self.disA2)
    loss_G_GAN2_B = self.backward_G_GAN(self.fake_B_random, self.disB2)
    if not self.no_ms:
      loss_G_GAN2_A2 = self.backward_G_GAN(self.fake_A_random2, self.disA2)
      loss_G_GAN2_B2 = self.backward_G_GAN(self.fake_B_random2, self.disB2)

    # mode seeking loss for A-->B and B-->A
    if not self.no_ms:
      lz_AB = torch.mean(torch.abs(self.fake_B_random2 - self.fake_B_random)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
      lz_BA = torch.mean(torch.abs(self.fake_A_random2 - self.fake_A_random)) / torch.mean(torch.abs(self.z_random2 - self.z_random))
      eps = 1 * 1e-5
      loss_lz_AB = 1 / (lz_AB + eps)
      loss_lz_BA = 1 / (lz_BA + eps)
    # latent regression loss
    if self.concat:
      loss_z_L1_a = torch.mean(torch.abs(self.mu2_a - self.z_random)) * 10
      loss_z_L1_b = torch.mean(torch.abs(self.mu2_b - self.z_random)) * 10
    else:
      loss_z_L1_a = torch.mean(torch.abs(self.z_attr_random_a - self.z_random)) * 10
      loss_z_L1_b = torch.mean(torch.abs(self.z_attr_random_b - self.z_random)) * 10

    loss_z_L1 = loss_z_L1_a + loss_z_L1_b + loss_G_GAN2_A + loss_G_GAN2_B

    if not self.no_ms:
      loss_z_L1 += (loss_G_GAN2_A2 + loss_G_GAN2_B2)
      loss_z_L1 += (loss_lz_AB + loss_lz_BA)
    loss_z_L1.backward()
    self.l1_recon_z_loss_a = loss_z_L1_a.item()
    self.l1_recon_z_loss_b = loss_z_L1_b.item()
    if not self.no_ms:
      self.gan2_loss_a = loss_G_GAN2_A.item() + loss_G_GAN2_A2.item()
      self.gan2_loss_b = loss_G_GAN2_B.item() + loss_G_GAN2_B2.item()
      self.lz_AB = loss_lz_AB.item()
      self.lz_BA = loss_lz_BA.item()
    else:
      self.gan2_loss_a = loss_G_GAN2_A.item()
      self.gan2_loss_b = loss_G_GAN2_B.item()

  def update_lr(self):
    self.disA_sch.step()
    self.disB_sch.step()
    self.disA2_sch.step()
    self.disB2_sch.step()
    self.disContent_sch.step()
    self.enc_c_sch.step()
    self.enc_c_shared_sch.step()
    self.enc_a_sch.step()
    self.gen_sch.step()
    if self.aux_masks:
      self.unet_dec_sch.step()

  def _l2_regularize(self, mu):
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def resume(self, model_dir, train=True):
    checkpoint = torch.load(model_dir, map_location=torch.device(f'cuda:{self.gpu}'))
    # weight
    if train:
      self.disA.load_state_dict(checkpoint['disA'])
      self.disA2.load_state_dict(checkpoint['disA2'])
      self.disB.load_state_dict(checkpoint['disB'])
      self.disB2.load_state_dict(checkpoint['disB2'])
      self.disContent.load_state_dict(checkpoint['disContent'])
    self.enc_c.load_state_dict(checkpoint['enc_c'])
    self.enc_c_shared.load_state_dict(checkpoint['enc_c_shared'])
    self.enc_a.load_state_dict(checkpoint['enc_a'])
    self.gen.load_state_dict(checkpoint['gen'])
    if self.aux_masks:
      self.unet_dec.load_state_dict(checkpoint['unet_dec'])
    # optimizer
    if train:
      self.disA_opt.load_state_dict(checkpoint['disA_opt'])
      self.disA2_opt.load_state_dict(checkpoint['disA2_opt'])
      self.disB_opt.load_state_dict(checkpoint['disB_opt'])
      self.disB2_opt.load_state_dict(checkpoint['disB2_opt'])
      self.disContent_opt.load_state_dict(checkpoint['disContent_opt'])
      self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
      self.enc_c_shared_opt.load_state_dict(checkpoint['enc_c_shared_opt'])
      self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
      self.gen_opt.load_state_dict(checkpoint['gen_opt'])
      if self.aux_masks:
        self.unet_dec_opt.load_state_dict(checkpoint['unet_dec_opt'])
    return checkpoint['ep'], checkpoint['total_it']

  def save(self, filename, ep, total_it):
    state = {
             'disA': self.disA.state_dict(),
             'disA2': self.disA2.state_dict(),
             'disB': self.disB.state_dict(),
             'disB2': self.disB2.state_dict(),
             'disContent': self.disContent.state_dict(),
             'enc_c': self.enc_c.state_dict(),
             'enc_c_shared': self.enc_c_shared.state_dict(),
             'enc_a': self.enc_a.state_dict(),
             'gen': self.gen.state_dict(),
             'unet_dec': self.unet_dec.state_dict() if self.aux_masks else None,
             'disA_opt': self.disA_opt.state_dict(),
             'disA2_opt': self.disA2_opt.state_dict(),
             'disB_opt': self.disB_opt.state_dict(),
             'disB2_opt': self.disB2_opt.state_dict(),
             'disContent_opt': self.disContent_opt.state_dict(),
             'enc_c_opt': self.enc_c_opt.state_dict(),
             'enc_c_shared_opt': self.enc_c_shared_opt.state_dict(),
             'enc_a_opt': self.enc_a_opt.state_dict(),
             'gen_opt': self.gen_opt.state_dict(),
             'unet_dec_opt': self.unet_dec_opt.state_dict() if self.aux_masks else None,
             'ep': ep,
             'total_it': total_it
              }
    torch.save(state, filename)
    return

  def assemble_outputs(self):
    images_a = self.normalize_image(self.real_A_encoded).detach().cpu()
    images_b = self.normalize_image(self.real_B_encoded).detach().cpu()
    images_a1 = self.normalize_image(self.fake_A_encoded).detach().cpu()
    images_a2 = self.normalize_image(self.fake_A_random).detach().cpu()
    images_a3 = self.normalize_image(self.fake_A_recon).detach().cpu()
    images_a4 = self.normalize_image(self.fake_AA_encoded).detach().cpu()
    images_b1 = self.normalize_image(self.fake_B_encoded).detach().cpu()
    images_b2 = self.normalize_image(self.fake_B_random).detach().cpu()
    images_b3 = self.normalize_image(self.fake_B_recon).detach().cpu()
    images_b4 = self.normalize_image(self.fake_BB_encoded).detach().cpu()

    if not self.aux_masks:
      row1 = torch.cat((images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]),3)
      row2 = torch.cat((images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]),3)
    else:
      images_a0 = self.normalize_image(self.A_mask_display).detach().cpu()
      images_b0 = self.normalize_image(self.B_mask_display).detach().cpu()
      row1 = torch.cat((images_a0[0:1, ::], images_a[0:1, ::], images_b1[0:1, ::], images_b2[0:1, ::], images_a4[0:1, ::], images_a3[0:1, ::]),3)
      row2 = torch.cat((images_b0[0:1, ::], images_b[0:1, ::], images_a1[0:1, ::], images_a2[0:1, ::], images_b4[0:1, ::], images_b3[0:1, ::]),3)

    return torch.cat((row1,row2),2)

  def normalize_image(self, x):
    return x[:,0:3,:,:]
