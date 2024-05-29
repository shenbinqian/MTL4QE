# -*- coding: utf-8 -*-

from src.loss_heuristics.solver import ProcrustesSolver
from src.loss_heuristics import basic_balancer
from src.loss_heuristics import balancers


@balancers.register("amtl")
class AlignedMTLBalancer(basic_balancer.BasicBalancer):
    def __init__(self, scale_mode='min', scale_decoder_grad=False, **kwargs):
        super().__init__(**kwargs)
        self.scale_decoder_grad = scale_decoder_grad
        self.scale_mode = scale_mode
        #print('AMGDA balancer scale mode:', self.scale_mode)

    def get_weighted_loss(self, losses,
             shared_params,
             task_specific_params
    ):
        grads = self.get_G_wrt_shared(losses, shared_params, update_decoder_grads=True)
        grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0), self.scale_mode)
        grad, weights = grads[0].sum(-1), weights.sum(-1)

        if self.compute_stats:
            self.compute_metrics(grads[0])

        self.set_shared_grad(shared_params, grad)
        if self.scale_decoder_grad is True:
            self.apply_decoder_scaling(task_specific_params, weights)

        weighted_loss = sum([losses[i] * weights[i] for i in range(len(weights))])

        return weighted_loss, weights
