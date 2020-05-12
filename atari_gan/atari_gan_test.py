import unittest

import gym
import numpy as np
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import nn, optim

from atari_gan import InputWrapper, Discriminator, Generator, iterate_batches, BATCH_SIZE
from atari_gan.generator import LATENT_VECTOR_SIZE

log = gym.logger
log.set_level(gym.logger.INFO)

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


class AtariGanTest(unittest.TestCase):
    def test_main(self):
        device = torch.device("cuda")
        envs = [
            InputWrapper(gym.make(name))
            for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')
        ]
        input_shape = envs[0].observation_space.shape

        net_discr = Discriminator(input_shape=input_shape).to(device)
        net_gener = Generator(output_shape=input_shape).to(device)

        objective = nn.BCELoss()
        gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
        writer = SummaryWriter()

        gen_losses = []
        dis_losses = []
        iter_no = 0

        true_labels_v = torch.ones(BATCH_SIZE, device=device)
        fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

        for batch_v in iterate_batches(envs):
            # fake samples, input is 4D: batch, filters, x, y
            gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
            gen_input_v.normal_(0, 1)
            gen_input_v = gen_input_v.to(device)
            batch_v = batch_v.to(device)
            gen_output_v = net_gener(gen_input_v)

            # train discriminator
            dis_optimizer.zero_grad()
            dis_output_true_v = net_discr(batch_v)
            dis_output_fake_v = net_discr(gen_output_v.detach())
            dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
            dis_loss.backward()
            dis_optimizer.step()
            dis_losses.append(dis_loss.item())

            # train generator
            gen_optimizer.zero_grad()
            dis_output_v = net_discr(gen_output_v)
            gen_loss_v = objective(dis_output_v, true_labels_v)
            gen_loss_v.backward()
            gen_optimizer.step()
            gen_losses.append(gen_loss_v.item())

            iter_no += 1
            if iter_no % REPORT_EVERY_ITER == 0:
                log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e", iter_no, np.mean(gen_losses), np.mean(dis_losses))
                writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
                writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
                gen_losses = []
                dis_losses = []
            if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
                writer.add_image("fake", vutils.make_grid(gen_output_v.data[:64], normalize=True), iter_no)
                writer.add_image("real", vutils.make_grid(batch_v.data[:64], normalize=True), iter_no)

    if __name__ == '__main__':
        unittest.main()
