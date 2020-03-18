import os
from importlib import import_module
import torch
import torch.nn as nn
from IPython import embed

class Model(nn.Module):
    def __init__(self, *args):
        super(Model, self).__init__()
        print('Making model...')

        self.args = args[0]
        self.ckp = args[1]
        self.crop = self.args.crop
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.precision = self.args.precision
        self.n_GPUs = self.args.n_GPUs
        self.save_models = self.args.save_models

        if self.args.model.find('DeepComp') >= 0:
            dc_type = self.args.model.split('-')[-1]
            module = import_module('model.deepcomp')
            self.model = module.make_model(self.args, dc_type)
        else:
            print('Import Module')
            module = import_module('model.' + self.args.model.lower())
            self.model = module.make_model(args)
        # if not next(self.model.parameters()).is_cuda:
        self.model = self.model.to(self.device)
        if self.args.precision == 'half': self.model = self.model.half()
        if not self.args.cpu:
            print('CUDA is ready!')
            torch.cuda.manual_seed(self.args.seed)
            if self.args.n_GPUs > 1:
                if not isinstance(self.model, nn.DataParallel):
                    self.model = nn.DataParallel(self.model, range(self.args.n_GPUs))

        # in the test phase of network pruning
        # if self.args.model.lower().find('prune') >= 0 and self.args.test_only:
        #     self.get_model().merge_conv()

        # not in the training phase of network pruning
        if not (self.args.model.lower().find('prune') >= 0 and not self.args.test_only and not self.args.load):
            self.load(
                self.ckp.dir,
                pretrained=self.args.pretrained,
                load=self.args.load,
                resume=self.args.resume,
                cpu=self.args.cpu
            )
        for m in self.modules():
            if hasattr(m, 'set_range'):
                m.set_range()

        print(self.get_model(), file=self.ckp.log_file)
        print(self.get_model())
        self.summarize(self.ckp)

    def forward(self, x):
        if self.crop > 1:
            b, n_crops, c, h, w = x.size()
            x = x.view(-1, c, h, w)
        x = self.model(x)

        if self.crop > 1: x = x.view(b, n_crops, -1).mean(1)

        return x

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        return self.get_model().state_dict(**kwargs)

    def save(self, apath, epoch, finetune=False, is_best=False):
        target = self.get_model().state_dict()

        conditions = (True, is_best, self.save_models)

        if finetune:
            names = ('finetune_latest', 'finetune_best', 'finetune_{}'.format(epoch))
        else:
            names = ('latest', 'best', '{}'.format(epoch))

        for c, n in zip(conditions, names):
            if c:
                torch.save(
                    target,
                    os.path.join(apath, 'model', 'model_{}.pt'.format(n))
                )

    def load(self, apath, pretrained='', load='', resume=-1, cpu=False):
        f = None
        if pretrained:
            if pretrained != 'download':
                print('Load pre-trained model from {}'.format(pretrained))
                f = pretrained
        else:
            if load:
                if resume == -1:
                    print('Load model after the last epoch')
                    resume = 'latest'
                elif resume == -2:
                    print('Load model after the last epoch in finetuning step')
                    resume = 'finetune_latest'
                else:
                    print('Load model after epoch {}'.format(resume))

                f = os.path.join(apath, 'model', 'model_{}.pt'.format(resume))

        if f:
            kwargs = {}
            if cpu:
                kwargs = {'map_location': lambda storage, loc: storage}

            state = torch.load(f, **kwargs)
            # embed()
            self.get_model().load_state_dict(state, strict=True)

    def begin(self, epoch, ckp):
        self.train()
        m = self.get_model()
        if hasattr(m, 'begin'):
            m.begin(epoch, ckp)

    def log(self, ckp):
        m = self.get_model()
        if hasattr(m, 'log'): m.log(ckp)

    def summarize(self, ckp):
        ckp.write_log('# parameters: {:,}'.format(sum([p.nelement() for p in self.model.parameters()])))

        kernels_1x1 = 0
        kernels_3x3 = 0
        kernels_others = 0
        gen = (c for c in self.model.modules() if isinstance(c, nn.Conv2d))
        for m in gen:
            kh, kw = m.kernel_size
            n_kernels = m.in_channels * m.out_channels
            if kh == 1 and kw == 1:
                kernels_1x1 += n_kernels
            elif kh == 3 and kw == 3:
                kernels_3x3 += n_kernels
            else:
                kernels_others += n_kernels

        linear = sum([l.weight.nelement() for l in self.model.modules()  if isinstance(l, nn.Linear)])

        ckp.write_log('1x1: {:,}\n3x3: {:,}\nOthers: {:,}\nLinear:{:,}\n'.
                      format(kernels_1x1, kernels_3x3, kernels_others, linear), refresh=True)

        if self.args.debug:
            def _get_flops(conv, x, y):
                _, _, h, w = y.size()
                kh, kw = conv.kernel_size
                conv.flops = h * w *conv.in_channels * conv.out_channels * kh * kw
                conv.flops_original = conv.flops

            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    m.register_forward_hook(_get_flops)

