# -*- coding: utf-8 -*-

import torch
import cvxpy as cp
import numpy as np
from typing import Dict, List, Tuple, Union
from src.loss_heuristics.aligned import AlignedMTLBalancer
from src.loss_heuristics.nash import NashMTL
from src.loss_heuristics.imtl import IMTLG


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        #torch.nn.init.orthogonal(m.weight)
        m.bias.data.fill_(0.01)


class LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.fc1(x)
        return x
'''

class LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, drop_rate=0.2):
        super(LinearLayer, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.LayerNorm(128),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(drop_rate),
            torch.nn.Linear(128, 32),
            torch.nn.LayerNorm(32),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(drop_rate),
            torch.nn.Linear(32, out_features))
        
        self.net.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        return x
'''


# Modify the forward pass to include the new layer
class ExtendedHead(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(ExtendedHead, self).__init__()
        self.pretrained_model = pretrained_model
        # deal with head1 and head2
        # input_size = self.pretrained_model.config.hidden_size
        self.sent_head = LinearLayer(in_features=2, out_features=1)
        self.word_head = LinearLayer(in_features=2, out_features=2)

    def combine_loss(self, sent_out, word_out, sent_scores, word_labels, alpha=1, beta=1, loss_type=None, evaluate=False):
        # sentence score loss
        sent_loss = torch.sum((sent_scores - sent_out.squeeze()) ** 2) / sent_scores.shape[0]

        # word-level classification loss
        num_labels = word_labels.shape[0] * word_labels.shape[1]
        # cross entropy loss
        criterion = torch.nn.CrossEntropyLoss()
        entropy_loss = criterion(word_out.view(-1,2), word_labels.view(-1))
        word_loss = entropy_loss / num_labels

        if evaluate:
            return sent_loss + word_loss, [0, 0]
        
        else:
            # return the combined loss
            if loss_type:
                # nash loss
                shared_parameters = list(self.pretrained_model.parameters())
                task_specific_params = {"sent_head": self.sent_head, "word_head": self.word_head}
                mtl_losses, weights = mtl_loss([sent_loss, word_loss], shared_parameters, task_specific_params, n_tasks=2, device=torch.device("cuda"), loss_type=loss_type)
                return torch.tensor(mtl_losses, requires_grad=True), weights
            else:
                return alpha * sent_loss + beta * word_loss, torch.tensor([alpha, beta]) #) / (alpha + beta), [alpha, beta]


    def forward(self, input_ids, attention_mask, sent_scores, word_labels, token_type_ids=None, alpha=1, beta=1, loss_type=None, evaluate=False, **kwargs):
        # Use the pretrained model for the base forward pass
        '''need to check the microtransquest achitecture and see how to extract the logits'''
        outputs = self.pretrained_model(input_ids, attention_mask, token_type_ids, output_hidden_states=True, **kwargs)
        
        # Extract the logits
        logits = outputs.logits
        #logits = outputs.hidden_states[-1]
        
        # sentence-level score
        sent_out = self.sent_head(logits[:,0,:])

        # word-level classification
        word_out = self.word_head(logits)
        

        loss, weights= self.combine_loss(sent_out, word_out, sent_scores, word_labels, alpha, beta, loss_type=loss_type, evaluate=evaluate)
        
        return loss, weights, sent_out, word_out


def mtl_loss(losses, shared_parameters, task_specific_params, n_tasks=2, device=torch.device("cuda"), loss_type="nash"):
    if loss_type == "nash":
        mtl = NashMTL(n_tasks=n_tasks, device=device)
        mtl_losses, weights = mtl.get_weighted_loss(losses, shared_parameters)
        return mtl_losses, weights
    elif loss_type == "imtlg":
        mtl = IMTLG()
        mtl_losses, weights = mtl.get_weighted_loss(losses, shared_parameters)
        return mtl_losses, weights
    elif loss_type == "dwa":
        mtl = DynamicWeightAverage(n_tasks=n_tasks, device=device)
        mtl_losses, weights = mtl.get_weighted_loss(losses)
        return mtl_losses, weights
    elif loss_type == "rlw":
        mtl = RLW(n_tasks=n_tasks, device=device)
        mtl_losses, weights = mtl.get_weighted_loss(losses)
        return mtl_losses, weights
    elif loss_type == "aligned":
        mtl = AlignedMTLBalancer()
        mtl_losses, weights = mtl.get_weighted_loss(losses, shared_parameters, task_specific_params)
        return mtl_losses, weights
    else:
        raise ValueError("loss_type must be either nash or imtlg")


class DynamicWeightAverage:
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Modification of: https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_split.py#L242
    """

    def __init__(
        self, n_tasks, device: torch.device, iteration_window: int = 25, temp=2.0
    ):
        """

        Parameters
        ----------
        n_tasks :
        iteration_window : 'iteration' loss is averaged over the last 'iteration_window' losses
        temp :
        """
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device
        self.iteration_window = iteration_window
        self.temp = temp
        self.running_iterations = 0
        self.costs = np.ones((iteration_window * 2, n_tasks), dtype=np.float32)
        self.weights = np.ones(n_tasks, dtype=np.float32)

    def get_weighted_loss(self, losses, **kwargs):
        losses = torch.tensor(losses, device=self.device)
        cost = losses.detach().cpu().numpy()

        # update costs - fifo
        self.costs[:-1, :] = self.costs[1:, :]
        self.costs[-1, :] = cost

        if self.running_iterations > self.iteration_window:
            ws = self.costs[self.iteration_window :, :].mean(0) / self.costs[
                : self.iteration_window, :
            ].mean(0)
            self.weights = (self.n_tasks * np.exp(ws / self.temp)) / (
                np.exp(ws / self.temp)
            ).sum()

        task_weights = torch.from_numpy(self.weights.astype(np.float32)).to(
            losses.device
        )
        loss = (task_weights * losses).mean()

        self.running_iterations += 1

        return loss, dict(weights=task_weights)
    

class RLW:
    """Random loss weighting: https://arxiv.org/pdf/2111.10603.pdf"""

    def __init__(self, n_tasks, device: torch.device):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs):
        assert len(losses) == self.n_tasks
        weight = (torch.nn.functional.softmax(torch.randn(self.n_tasks), dim=-1)).to(self.device)
        losses = torch.tensor(losses, device=self.device)
        loss = torch.sum(losses * weight)

        return loss, dict(weights=weight)