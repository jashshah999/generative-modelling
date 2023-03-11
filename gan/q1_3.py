import argparse
import os
from utils import get_args
from torch.autograd import Variable
import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp=None, interp=None, lamb=None
):
    """
    TODO 1.3.1: Implement GAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    # """
    criterion = torch.nn.BCEWithLogitsLoss()
    discrim_real_ones = torch.full_like(discrim_real,1.0,dtype=torch.float).cuda()
    discrim_fake_zeros = torch.full_like(discrim_real,0.0,dtype=torch.float).cuda()
    true = torch.zeros_like(discrim_real,dtype=torch.float)
    false = torch.ones_like(discrim_fake,dtype=torch.float)
    loss1 = F.binary_cross_entropy_with_logits(discrim_real, discrim_real_ones)
    loss2 = F.binary_cross_entropy_with_logits(discrim_fake, discrim_fake_zeros)
    loss = (loss1 +loss2)/2

    # loss = 0
    # discrim_real_ones = torch.full_like(discrim_real,1.0,dtype=torch.float).cuda()
    # loss += criterion(discrim_real, discrim_real_ones)

    # discrim_fake_zeros = torch.full_like(discrim_real,0.0,dtype=torch.float).cuda()
    # loss += criterion(discrim_fake, discrim_fake_zeros)
    # print("Disc_Loss:",loss)
    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.3.1: Implement GAN loss for generator.
    
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    discrim_fake_ones = torch.full_like(discrim_fake,1.0,dtype=torch.float).cuda()
    fool = torch.ones_like(discrim_fake,dtype=torch.float)
    loss = F.binary_cross_entropy_with_logits(discrim_fake, discrim_fake_ones)
    # criterion = torch.nn.BCEWithLogitsLoss()
    # loss=0
    # discrim_fake_ones = torch.full_like(discrim_fake,1.0,dtype=torch.float).cuda()
    # loss += criterion(discrim_fake, discrim_fake_ones)
    # print("Gen_Loss:",loss)
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    # gen = Generator()
    # disc= Discriminator()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )

    # print(gen)
    # print(disc)