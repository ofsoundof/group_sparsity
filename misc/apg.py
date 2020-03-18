from torch.optim import SGD
import torch
from IPython import embed

def proximal_operator_l1(param, i, regularization, lr):
        ps = param.shape
        p = param.squeeze().t()
        eps = 1e-6
        if i % 2 == 0:
            n = torch.norm(p, p=2, dim=0)
            scale = torch.max(1 - regularization * lr / (n + eps), torch.zeros_like(n, device=n.device))
            scale = scale.repeat(ps[0], 1)
            if torch.isnan(n[0]):
                embed()
        else:
            n = torch.norm(p, p=2, dim=1)
            scale = torch.max(1 - regularization * lr / (n + eps), torch.zeros_like(n, device=n.device))
            scale = scale.repeat(ps[0], 1).t()
        # p = param.data
        # scale = torch.ones(ps).to(param.device) * 0.9
        # print(scale)
        return torch.mul(scale, p).t().view(ps)

class APG(SGD):

    def __init__(self, params, lr, regularization, prox_frequency, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        super(APG, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                                  weight_decay=weight_decay, nesterov=nesterov)
        self.regularization = regularization
        self.converging = False
        self.prox_frequency = prox_frequency
        self.batch = 1

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for i, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            if self.converging or (not self.converging and i == 0) or \
                    (not self.converging and i == 1 and self.batch % self.prox_frequency != 0):
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)
            elif not self.converging and i == 1 and self.batch % self.prox_frequency == 0:
                for j, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    momentum = 0.1
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            z = p.data - group['lr'] * p.grad.data  #eqn 10
                            z = proximal_operator_l1(z, j, regularization=self.regularization, lr=group['lr']) #soft thresholding
                            buf.mul_(momentum).add_(z - p.data)     #eqn 11
                            p.data = z + momentum * buf             #eqn 12
                    else:
                        p.data.add_(-group['lr'], d_p)

        return loss
