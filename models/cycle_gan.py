import torch.nn as nn
import torch
from torch.optim import lr_scheduler
from . import networks


def set_grads(required, notrequired):
    for net in required:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = True
    for net in notrequired:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = False


def loss_GAN(prediction, target, reduction='mean'):
    loss = nn.MSELoss(reduction=reduction)
    # loss = nn.BCEWithLogitsLoss()
    return loss(prediction, torch.full_like(prediction, target))


def loss_GAN_D(G_x, D_y):
    return 0.5 * (loss_GAN(G_x, 0) + loss_GAN(D_y, 1))


def loss_GAN_G(G_x):
    return loss_GAN(G_x, 1)


def cycle_consistency_loss(generated, original):
    loss = nn.L1Loss()
    return loss(generated, original)


class CycleGAN():
    def __init__(self, opt) -> None:
        self.rec_X = None
        self.rec_Y = None
        self.fake_X = None
        self.fake_Y = None
        self.input_Y = None
        self.input_X = None
        self.opt = opt
        self.criterionIdt = torch.nn.L1Loss()
        self.generator_X2Y = networks.GeneratorResnet(opt.d_input_nc, opt.d_input_nc, ngf=64, use_dropout=False,
                                                      n_blocks=6).to(torch.device('cuda'))
        self.generator_Y2X = networks.GeneratorResnet(opt.d_input_nc, opt.d_input_nc, ngf=64, use_dropout=False,
                                                      n_blocks=6).to(torch.device('cuda'))
        self.discriminator_X = networks.Discriminator(
            opt.g_input_nc, ndf=64, n_layers=3).to(torch.device('cuda'))
        self.discriminator_Y = networks.Discriminator(
            opt.g_input_nc, ndf=64, n_layers=3).to(torch.device('cuda'))

        self.optimizer_G = torch.optim.Adam(
            list(self.generator_X2Y.parameters()) + list(self.generator_Y2X.parameters()), lr=opt.lr_g)
        self.optimizer_D = torch.optim.Adam(
            list(self.discriminator_X.parameters()) + list(self.discriminator_Y.parameters()), lr=opt.lr_d)

        self.scheduler_g = lr_scheduler.StepLR(
            self.optimizer_G, step_size=opt.lr_decay_iters, gamma=0.1)

        self.scheduler_d = lr_scheduler.StepLR(
            self.optimizer_D, step_size=opt.lr_decay_iters, gamma=0.1)

    def forward(self, input_X, input_Y):
        self.input_X = input_X
        self.input_Y = input_Y
        self.fake_Y = self.generator_X2Y(input_X)
        self.fake_X = self.generator_Y2X(input_Y)
        self.rec_Y = self.generator_X2Y(self.fake_X)
        self.rec_X = self.generator_Y2X(self.fake_Y)

    def backward_D_X(self):
        pre_real_X = self.discriminator_X(self.input_X)
        pre_fake_X = self.discriminator_X(self.fake_X.detach())
        loss = loss_GAN_D(pre_fake_X, pre_real_X)
        loss.backward()
        return loss

    def backward_D_Y(self):
        pre_real_Y = self.discriminator_Y(self.input_Y)
        pre_fake_Y = self.discriminator_Y(self.fake_Y.detach())
        loss = loss_GAN_D(pre_fake_Y, pre_real_Y)
        loss.backward()
        return loss

    def loss_G_X2Y(self):
        pre_fake_Y = self.discriminator_Y(self.fake_Y)
        return loss_GAN_G(pre_fake_Y) + self.opt.lamnda_X2Y * cycle_consistency_loss(self.rec_X, self.input_X)

    def loss_G_Y2X(self):
        pre_fake_X = self.discriminator_X(self.fake_X)
        return loss_GAN_G(pre_fake_X) + self.opt.lamnda_Y2X * cycle_consistency_loss(self.rec_Y, self.input_Y)

    def backward_G(self):
        if self.opt.lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.generator_X2Y(self.input_Y)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.input_Y) * self.opt.lamnda_X2Y * self.opt.lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.generator_Y2X(self.input_X)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.input_X) * self.opt.lamnda_X2Y * self.opt.lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        
        loss = self.loss_G_X2Y() + self.loss_G_Y2X() + self.loss_idt_A + self.loss_idt_B
        loss.backward()
        return loss

    def generate_X(self, input_Y):
        return self.generator_Y2X(input_Y)

    def generate_Y(self, input_X):
        return self.generator_X2Y(input_X)

    def train(self, input_X, input_Y):
        self.forward(input_X, input_Y)
        # set_grads([self.generator_X2Y, self.generator_Y2X], [self.discriminator_X, self.discriminator_Y])

        self.optimizer_G.zero_grad()
        loss_g = self.backward_G()
        self.optimizer_G.step()

        # set_grads([self.discriminator_X, self.discriminator_Y], [self.generator_X2Y, self.generator_Y2X])

        self.optimizer_D.zero_grad()

        loss_d_y = self.backward_D_Y()
        loss_d_x = self.backward_D_X()

        self.optimizer_D.step()

        return loss_g.item(), loss_d_x.item(), loss_d_y.item()

    def update_lr(self):
        self.scheduler_g.step()
        self.scheduler_d.step()

    def save_model(self, mode, path, type = 'lowest'):
        if mode == 'X':
            torch.save(self.generator_Y2X, path + '/models/' + type + '_G_Y2X.pth')
            print(f"Generotor G_Y2X saved![" + type + ']')
        else:
            torch.save(self.generator_X2Y, path + '/models/' + type + '_G_X2Y.pth')
            print(f"Generotor G_X2Y saved![" + type + ']')
