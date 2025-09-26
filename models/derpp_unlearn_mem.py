# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from models import register_model
from utils.args import add_rehearsal_args, ArgumentParser
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import torch
import copy
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
epsilon = 1E-20




@register_model('derpp-unlearn-mem')
class DerppUnlearnMem(ContinualModel):
    NAME = 'derpp_unlearn_mem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')

        # Unlearning parameters
        parser.add_argument('--delta', type=float, default=0.00001,
                            help='Unlearning rate for plasticity enhancement')
        parser.add_argument('--tau', type=float, default=0.00001,
                            help='Fisher information decay rate')
        parser.add_argument('--unlearn_frequency', type=int, default=1,
                            help='How often to perform unlearning (every N steps)')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(DerppUnlearnMem, self).__init__(backbone, loss, args, transform, dataset=dataset)

        # Initialize simple reservoir buffer
        self.buffer = Buffer(self.args.buffer_size)

        self.temp = copy.deepcopy(self.net).to(self.device)
        self.temp_opt = torch.optim.SGD(self.temp.parameters(), lr=0.01)

        lr = self.args.lr
        weight_decay = 0.0001

        # Unlearning parameters
        self.delta = getattr(args, 'delta', 0.00001)
        self.tau = getattr(args, 'tau', 0.00001)
        self.unlearn_frequency = getattr(args, 'unlearn_frequency', 1)

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = {}
        for name, param in self.net.named_parameters():
            self.fish[name] = torch.zeros_like(param).to(self.device)

        # Initialize step counter for unlearning frequency
        self.step_counter = 0

        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def observe(self, inputs, labels, not_aug_inputs):
        # Increment step counter
        self.step_counter += 1

        # Unlearn on buffer data to enhance plasticity
        if not self.buffer.is_empty() and self.step_counter % self.unlearn_frequency == 0:
            buf_inputs_unlearn, buf_labels_unlearn, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            self.unlearn(inputs=buf_inputs_unlearn, labels=buf_labels_unlearn)

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            # Standard DER++ buffer replay with logits and classification loss
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        # Add data to buffer
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()

    def unlearn(self, inputs, labels):
        """
        Unlearning mechanism to enhance plasticity by learning in reverse direction
        """
        self.temp.load_state_dict(self.net.state_dict())
        self.temp.train()
        outputs = self.temp(inputs)
        loss = - F.cross_entropy(outputs, labels)
        self.temp_opt.zero_grad()
        loss.backward()
        self.temp_opt.step()

        for (model_name, model_param), (temp_name, temp_param) in zip(self.net.named_parameters(), self.temp.named_parameters()):
            weight_update = temp_param - model_param
            model_param_norm = model_param.norm()
            weight_update_norm = weight_update.norm() + epsilon
            norm_update = model_param_norm / weight_update_norm * weight_update
            identity = torch.ones_like(self.fish[model_name])
            with torch.no_grad():
                model_param.add_(self.delta * torch.mul(1.0/(identity + 0.001*self.fish[model_name]), norm_update + 0.001*torch.randn_like(norm_update)))

    def end_task(self, dataset):
        """
        Calculate Fisher Information Matrix on buffer data to protect old knowledge
        """
        self.temp.load_state_dict(self.net.state_dict())
        fish = {}
        for name, param in self.temp.named_parameters():
            fish[name] = torch.zeros_like(param).to(self.device)

        # Calculate FIM on buffer data
        if not self.buffer.is_empty():
            all_buffer_data = self.buffer.get_all_data()
            buf_examples = all_buffer_data[0]
            buf_labels = all_buffer_data[1]

            buf_dataset = TensorDataset(buf_examples, buf_labels)
            buf_loader = DataLoader(buf_dataset, batch_size=self.args.batch_size)

            for inputs, labels in buf_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                for ex, lab in zip(inputs, labels):
                    self.temp_opt.zero_grad()
                    output = self.temp(ex.unsqueeze(0))
                    loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                          reduction='none')
                    exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                    loss = torch.mean(loss)
                    loss.backward()

                    for name, param in self.temp.named_parameters():
                        if param.grad is not None:
                            fish[name] += exp_cond_prob * param.grad ** 2

            # Normalize FIM by buffer size
            for name, param in self.temp.named_parameters():
                fish[name] /= len(self.buffer)

        # Update global FIM
        for key in self.fish:
            self.fish[key] *= self.tau
            if key in fish:
                self.fish[key] += fish[key].to(self.device)

        self.checkpoint = self.net.get_params().data.clone()
        self.temp_opt.zero_grad()