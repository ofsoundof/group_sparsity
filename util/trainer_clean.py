# This module is used to train the original network instead of pruning or decomposition.
from util import utility
import torch
import torch.nn as nn
from tqdm import tqdm
from model.in_use.flops_counter import get_model_flops
import matplotlib
matplotlib.use('Agg')


# torch.autograd.set_detect_anomaly(True)


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, writer=None):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.writer = writer

        if args.data_train.find('CIFAR') >= 0:
            self.input_dim = (3, 32, 32)
        elif args.data_train.find('Tiny_ImageNet') >= 0:
            self.input_dim = (3, 64, 64)
        else:
            self.input_dim = (3, 224, 224)
        #set_output_dimension(self.model.get_model(), self.input_dim)
        #self.flops = get_flops(self.model.get_model())
        #self.params = get_parameters(self.model.get_model())
        #self.ckp.write_log('\nThe computation complexity and number of parameters of the current network is as follows.'
        #                   '\nFlops: {:.4f} [G]\tParams {:.2f} [k]'.format(self.flops / 10. ** 9, self.params / 10. ** 3))
        self.flops_another = get_model_flops(self.model.get_model(), self.input_dim, False)
        self.ckp.write_log('Flops: {:.4f} [G] calculated by the original counter. \nMake sure that the two calculated '
                           'Flops are the same.\n'.format(self.flops_another / 10. ** 9))

        self.optimizer = utility.make_optimizer(args, self.model, ckp=ckp)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.model.find('INQ') >= 0:
            self.inq_steps = args.inq_steps
        else:
            self.inq_steps = None

    def train(self):
        epoch, _ = self.start_epoch()
        self.model.begin(epoch, self.ckp) #TODO: investigate why not using self.model.train() directly
        self.loss.start_log()

        timer_data, timer_model = utility.timer(), utility.timer()
        n_samples = 0

        for batch, (img, label) in enumerate(self.loader_train):
            # if batch<=1:
            img, label = self.prepare(img, label)
            n_samples += img.size(0)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            # embed()
            prediction = self.model(img)
            loss, _ = self.loss(prediction, label)

            loss.backward()
            self.optimizer.step()

            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(
                    '{}/{} ({:.0f}%)\t'
                    'NLL: {:.3f}\t'
                    'Top1: {:.2f} / Top5: {:.2f}\t'
                    'Time: {:.1f}+{:.1f}s'.format(
                        n_samples,
                        len(self.loader_train.dataset),
                        100.0 * n_samples / len(self.loader_train.dataset),
                        *(self.loss.log_train[-1, :] / n_samples),
                        timer_model.release(),
                        timer_data.release()
                    )
                )

            if self.args.summary:
                if (batch + 1) % 50 == 0:
                    for name, param in self.model.named_parameters():
                        if name.find('features') >= 0 and name.find('weight') >= 0:
                            self.writer.add_scalar('data/' + name, param.clone().cpu().data.abs().mean().numpy(),
                                                   1000 * (epoch - 1) + batch)
                            self.writer.add_scalar('data/' + name + '_grad', param.grad.clone().cpu().data.abs().mean().numpy(),
                                                   1000 * (epoch - 1) + batch)

                if (batch + 1) == 500:
                    for name, param in self.model.named_parameters():
                        if name.find('features') >= 0 and name.find('weight') >= 0:
                            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), 1000 * (epoch - 1) + batch)
                            self.writer.add_histogram(name + '_grad', param.grad.clone().cpu().data.numpy(),
                                                      1000 * (epoch - 1) + batch)
            # else:
            #     break

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

                # if self.args.debug:
                #     self._analysis()

        self.loss.end_log(len(self.loader_test.dataset), train=False)

        # Lower is better
        best = self.loss.log_test.min(0)
        for i, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
            self.ckp.write_log(
                '{}: {:.3f} (Best: {:.3f} from epoch {})'.format(
                    measure,
                    self.loss.log_test[-1, i],
                    best[0][i],
                    best[1][i] + 1
                )
            )

        self.ckp.write_log(
            'Time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        is_best = self.loss.log_test[-1, self.args.top] <= best[0][self.args.top]
        self.ckp.save(self, epoch, is_best=is_best)
        self.ckp.save_results(epoch, self.model)

        # scheduler.step is moved from training procedure to test procedure
        self.scheduler.step()

    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()
            return x

        return [_prepare(a) for a in args]

    def start_epoch(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2}'.format(epoch, lr))

        return epoch, lr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

    def _analysis(self):
        flops = torch.Tensor([getattr(m, 'flops', 0) for m in self.model.modules()])
        flops_conv = torch.Tensor([getattr(m, 'flops', 0) for m in self.model.modules() if isinstance(m, nn.Conv2d)])
        flops_ori = torch.Tensor([getattr(m, 'flops_original', 0) for m in self.model.modules()])

        print('')
        print('FLOPs: {:.2f} x 10^8'.format(flops.sum() / 10**8))
        print('Compressed: {:.2f} x 10^8 / Others: {:.2f} x 10^8'.format(
            (flops.sum() - flops_conv.sum()) / 10**8 , flops_conv.sum() / 10**8
        ))
        print('Accel - Total original: {:.2f} x 10^8 ({:.2f}x)'.format(
            flops_ori.sum() / 10**8, flops_ori.sum() / flops.sum()
        ))
        print('Accel - 3x3 original: {:.2f} x 10^8 ({:.2f}x)'.format(
            (flops_ori.sum() - flops_conv.sum()) / 10**8,
            (flops_ori.sum() - flops_conv.sum()) / (flops.sum() - flops_conv.sum())
        ))
        input()

