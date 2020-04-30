import torch
import matplotlib
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from util import utility
from model_hinge.hinge_utility import reg_anneal
from loss import distillation
matplotlib.use('Agg')
#from IPython import embed


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, writer=None, converging=False, model_teacher=None):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.model_teacher = model_teacher
        self.loss = my_loss
        self.converging = converging
        self.writer = writer
        self.lr_adjust_flag = self.args.model.lower().find('resnet') >= 0 #TODO: print rP

        self.optimizer = utility.make_optimizer_hinge(args, self.model, ckp, self.converging, self.lr_adjust_flag)
        self.scheduler = utility.make_scheduler_hinge(args, self.optimizer, self.converging, self.lr_adjust_flag)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.model.find('INQ') >= 0:
            self.inq_steps = args.inq_steps
        else:
            self.inq_steps = None

    def reset_after_optimization(self, epoch_continue):
        if not self.converging and not self.args.test_only:
            self.converging = True
            # In Phase 1 & 3, the optimizer and scheduler are reset.
            # In Phase 2, the optimizer and scheduler is not used.
            # In Phase 4, the optimizer and scheduler is already set during the initialization of the trainer.
            # during the converging stage, self.converging =True. Do not need to set lr_adjust_flag in make_optimizer_hinge
            #   and make_scheduler_hinge.
            self.optimizer = utility.make_optimizer_hinge(self.args, self.model, self.ckp, self.converging)
            self.scheduler = utility.make_scheduler_hinge(self.args, self.optimizer, self.converging)
        if not self.args.test_only and self.args.summary:
            self.writer = SummaryWriter(os.path.join(self.args.dir_save, self.args.save), comment='converging')
        self.epoch_continue = epoch_continue

    def train(self):
        epoch = self.start_epoch()
        self.model.begin(epoch, self.ckp) #TODO: investigate why not using self.model.train() directly
        self.loss.start_log()
        # modules = self.model.get_model().find_modules() #TODO: merge this
        timer_data, timer_model = utility.timer(), utility.timer()
        n_samples = 0

        for batch, (img, label) in enumerate(self.loader_train):
            img, label = self.prepare(img, label)
            n_samples += img.size(0)

            timer_data.hold()
            timer_model.tic()

            # Forward pass and computing the loss function
            self.optimizer.zero_grad()
            prediction = self.model(img)
            loss, _ = self.loss(prediction, label)
            lossp = self.model.get_model().compute_loss(batch + 1, epoch, self.converging)
            if not self.converging:
                # use projection loss for SGD and don't use it for PG
                if self.args.optimizer == 'SGD':
                    loss = loss + sum(lossp)
            else:
                # use distillation loss
                if self.args.distillation:
                    with torch.no_grad():
                        prediction_teacher = self.model_teacher(img)
                    loss_distill = distillation(prediction, prediction_teacher, T=4)
                    loss = loss_distill * 0.4 + loss * 0.6
            # Backward pass and computing the gradients
            loss.backward()
            # Update learning rate based on the gradients. ResNet20, 56, 164, and Wide ResNet

            if not self.converging and self.lr_adjust_flag:
                self.model.get_model().update_grad_ratio()
                self.scheduler.running_grad_ratio = self.model.get_model().running_grad_ratio
                for param_group, lr in zip(self.optimizer.param_groups, self.scheduler.get_lr()):
                    param_group['lr'] = lr

            # Update the parameters
            if self.args.optimizer == 'SGD':
                self.optimizer.step()
            elif self.args.optimizer == 'PG':
                # Gradient step
                self.optimizer.step()
                if not self.converging and (batch + 1) % self.args.prox_freq == 0:
                    # Anneal the regularization factor
                    reg = reg_anneal(lossp[0], self.args.regularization_factor, self.args.annealing_factor,
                                     self.args.annealing_t1, self.args.annealing_t2)
                    # Proximal step
                    self.model.get_model().proximal_operator(self.scheduler.get_lr()[-1], batch+1, reg)
            elif self.args.optimizer == 'APG': # TODO: still interesting to investigate APG
                self.optimizer.converging = self.converging
                self.optimizer.batch = batch + 1
                self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                s = '{}/{} ({:.0f}%)\tTotal: {:.3f} / P1: {:.3f}'.\
                    format(n_samples, len(self.loader_train.dataset),
                           100.0 * n_samples / len(self.loader_train.dataset), loss, lossp[0])
                if len(lossp) == 2:
                    s += ' / P2: {:.3f}'.format(lossp[1])
                if not self.converging:
                    if self.lr_adjust_flag:
                        s += ' / rP: {:.3f}'.format(self.model.get_model().running_grad_ratio)
                else:
                    if self.args.distillation:
                        s += ' / Dis: {:.3f}'.format(loss_distill)
                s += ' / NLL: {:.3f}\tTop1: {:.2f} / Top5: {:.2f}\tTime: {:.1f}+{:.1f}s'.\
                    format(*(self.loss.log_train[-1, :] / n_samples), timer_model.release(), timer_data.release())
                self.ckp.write_log(s)

            if self.args.summary:
                if (batch + 1) % 50 == 0:
                    for name, param in self.model.named_parameters():
                        if name.find('features') >= 0 and name.find('weight') >= 0:
                            self.writer.add_scalar('data/' + name, param.clone().cpu().data.abs().mean().numpy(),
                                                   1000 * (epoch - 1) + batch)
                            if param.grad is not None:
                                self.writer.add_scalar('data/' + name + '_grad',
                                                       param.grad.clone().cpu().data.abs().mean().numpy(),
                                                       1000 * (epoch - 1) + batch)
                if (batch + 1) == 500:
                    for name, param in self.model.named_parameters():
                        if name.find('features') >= 0 and name.find('weight') >= 0:
                            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), 1000 * (epoch - 1) + batch)
                            if param.grad is not None:
                                self.writer.add_histogram(name + '_grad', param.grad.clone().cpu().data.numpy(),
                                                          1000 * (epoch - 1) + batch)

            timer_data.tic()
        self.model.log(self.ckp)
        self.loss.end_log(len(self.loader_train.dataset))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.loss.start_log(train=False)
        self.model.eval()

        timer_test = utility.timer()
        timer_test.tic()
        with torch.no_grad():
            for img, label in tqdm(self.loader_test, ncols=80):
                img, label = self.prepare(img, label)
                prediction = self.model(img)
                self.loss(prediction, label, train=False)

        self.loss.end_log(len(self.loader_test.dataset), train=False)

        # Lower is better
        best = self.loss.log_test.min(0)
        for i, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
            self.ckp.write_log('{}: {:.3f} (Best: {:.3f} from epoch {})'.
                               format(measure, self.loss.log_test[-1, i], best[0][i], best[1][i] + 1))

        if hasattr(self, 'epoch_continue') and self.converging:
            best = self.loss.log_test[:self.epoch_continue, :].min(0)
            self.ckp.write_log('\nBest during searching')
            for i, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
                self.ckp.write_log('{}: {:.3f} from epoch {}'.format(measure, best[0][i], best[1][i]))
        self.ckp.write_log('Time: {:.2f}s\n'.format(timer_test.toc()), refresh=True)

        is_best = self.loss.log_test[-1, self.args.top] <= best[0][self.args.top]
        self.ckp.save(self, epoch, converging=self.converging, is_best=is_best)
        # This is used by clustering convolutional kernels
        # self.ckp.save_results(epoch, self.model)

        # scheduler.step is moved from training procedure to test procedure
        self.scheduler.step()

        # modules = [m for m in self.model.get_model().modules() if isinstance(m, ResBlock)]
        # from model.prune_utility import print_array_on_one_line
        # for i, m in enumerate(modules):
        #     if i in [4, 8, 13, 17, 22, 26]:
        #         w1 = m._modules['body']._modules['0']._modules['1'].weight.squeeze().t()
        #         n1 = torch.norm(w1, p=2, dim=0)
        #         w2 = m._modules['body']._modules['3']._modules['1'].weight.squeeze().t()
        #         n2 = torch.norm(w2, p=2, dim=1)
        #         with print_array_on_one_line():
        #             print('Norm of Projection1 {}'.format(n1.detach().cpu().numpy()))
        #             print('Norm of Projection2 {}\n'.format(n2.detach().cpu().numpy()))

    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()
            return x

        return [_prepare(a) for a in args]

    def start_epoch(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()
        if len(lr) == 1:
            s = '[Epoch {}]\tLearning rate: {:.2}'.format(epoch, lr[0])
        else:
            s = '[Epoch {}]\tLearning rate:'.format(epoch)
            for i, l in enumerate(lr):
                s += ' Group {} - {:.2}'.format(i, l) if i + 1 == len(lr) else ' Group {} - {:.2} /'.format(i, l)

        if not self.converging:
            stage = 'Searching Stage'
        else:
            stage = 'Converging Stage (Searching Epoch {})'.format(self.epoch_continue)
        s += '\t{}'.format(stage)
        self.ckp.write_log(s)
        return epoch

    def terminate(self):
        if self.args.test_only:
            # if self.args.model.lower().find('hinge') >= 0:
            #     self.model.get_model().compress()
            #     if self.args.model.lower() in ['hinge_resnet56', 'hinge_wide_resnet', 'hinge_densenet']:
            #         self.model.get_model().merge_conv()
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            if not self.converging:
                return epoch > 200
            else:
                return epoch > self.args.epochs


def proximal_operator_l0(optimizer, regularization, lr):
    for i, param in enumerate(optimizer.param_groups[1]['params']):
        ps = param.data.shape
        p = param.data.squeeze().t()
        eps = 1e-6
        if i % 2 == 0:
            n = torch.norm(p, p=2, dim=0)
            scale = (n > regularization).to(torch.float32)
            scale = scale.repeat(ps[0], 1)
            if torch.isnan(n[0]):
                embed()
        else:
            n = torch.norm(p, p=2, dim=1)
            scale = (n > regularization).to(torch.float32)
            scale = scale.repeat(ps[0], 1).t()
        # p = param.data
        # scale = torch.ones(ps).to(param.device) * 0.9
        param.data = torch.mul(scale, p).t().view(ps)


def proximal_operator_l1(optimizer, regularization, lr):
    for i, param in enumerate(optimizer.param_groups[1]['params']):
        ps = param.data.shape
        p = param.data.squeeze().t()
        eps = 1e-6
        if i % 2 == 0:
            n = torch.norm(p, p=2, dim=0)
            scale = torch.max(1 - regularization * lr / (n + eps), torch.zeros_like(n, device=n.device))
            # scale = scale.repeat(ps[0], 1)
            scale = scale.repeat(ps[1], 1)
            if torch.isnan(n[0]):
                embed()
        else:
            n = torch.norm(p, p=2, dim=1)
            scale = torch.max(1 - regularization * lr / (n + eps), torch.zeros_like(n, device=n.device))
            scale = scale.repeat(ps[0], 1).t()
        # p = param.data
        # scale = torch.ones(ps).to(param.device) * 0.9
        param.data = torch.mul(scale, p).t().view(ps)
