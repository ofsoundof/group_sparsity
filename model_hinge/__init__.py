import torch
import torch.nn as nn
import os
from importlib import import_module
# from IPython import embed


def teacher_model(model):
    # Determine the teacher model for knowledge distillation.
    model = model.lower()
    if model.find('densenet') >= 0:
        model = 'densenet'
    elif model.find('resnet') >= 0:
        model = 'resnet'
    elif model.find('resnet_bottleneck') >= 0:
        model = 'resnet'
    elif model.find('resnext') >= 0:
        model = 'resnext'
    elif model.find('vgg') >= 0:
        model = 'vgg'
    elif model.find('wide_resnet') >= 0:
        model = 'wide_resnet'
    else:
        raise NotImplementedError('Compressing model {} is not implemented'.format(model))
    return  model


class Model(nn.Module):
    def __init__(self, args, checkpoint, converging=False, teacher=False):
        """
        :param args:
        :param checkpoint:
        :param converging: needed to decide whether to load the optimization or the finetune model
        :param teacher: indicate whether this is a teacher model used in knowledge distillation
        """
        super(Model, self).__init__()
        print('Making model...')

        self.args = args
        self.ckp = checkpoint
        self.crop = args.crop
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.precision = args.precision
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        print('Import Module')
        if not teacher:
            model = args.model
            pretrain = args.pretrain
            module = import_module('model_hinge.' + model.lower())
            self.model = module.make_model(args, checkpoint, converging)
        else:
            model = teacher_model(args.model)
            pretrain = args.teacher
            module = import_module('model.' + model.lower())
            self.model = module.make_model([args])
        self.model = self.model.to(self.device)
        if args.precision == 'half':
            self.model = self.model.half()
        if not args.cpu:
            print('CUDA is ready!')
            torch.cuda.manual_seed(args.seed)
            if args.n_GPUs > 1:
                if not isinstance(self.model, nn.DataParallel):
                    self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        # in the test phase of network compression
        if args.test_only:
            if pretrain.find('merge') >= 0:
                self.get_model().merge_conv()

        self.load(pretrain, args.load, args.resume, args.cpu, converging, args.test_only, teacher)

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

    def save(self, apath, epoch, converging=False, is_best=False):
        target = self.get_model().state_dict()

        conditions = (True, is_best, self.save_models)

        if converging:
            names = ('converging_latest', 'converging_best', 'converging_{}'.format(epoch))
        else:
            names = ('latest', 'best', '{}'.format(epoch))

        for c, n in zip(conditions, names):
            if c:
                torch.save(
                    target,
                    os.path.join(apath, 'model', 'model_{}.pt'.format(n))
                )

    def load(self, pretrain='', load='', resume=-1, cpu=False, converging=False, test_only=False, teacher=False):
        """
        Use pretrain and load to determine how to load the model.
        For 'Group Sparsity: the Hinge (CVPR2020)', pretrain is always set, namely, not empty string.
        1. Phase 1, load == '', test_only = False. Training phase, the model is not loaded here (in the hinge functions).
        2. Phase 2, load == '', test_only = True. Testing phase.
        3. Phase 3, load != '', converging = False, loading for searching, the model is also loaded in the hinge functions.
        4. Phase 4, load != '', converging = True, loading for converging, the model is not loaded in the hinge functions.
        """
        if not teacher:
            if not load:
                if not test_only:
                    # Phase 1, training phase, the model is not loaded in the hinge_** functions instead of here .
                    f = None
                    print('During training phase, the model is loaded here.')
                else:
                    # Phase 2, testing phase, loading the pruned model, strict = False.
                    f = os.path.join(pretrain, 'model/model_latest.pt') if pretrain.find('.pt') < 0 else pretrain
                    print('Load pre-trained model from {}'.format(f))
                    strict = False
            else:
                if not converging:
                    # Phase 3, loading in the searching stage, loading the unpruned model, strict =True
                    if resume == -1:
                        print('Load model after the last epoch')
                        resume = 'latest'
                    else:
                        print('Load model after epoch {}'.format(resume))
                    strict = True
                else:
                    # Phase 4, loading in the converging stage, loading the pruned model, strict = False
                    if resume == -1:
                        print('Load model after the last epoch in converging stage')
                        resume = 'converging_latest'
                    else:
                        print('Load model after epoch {} in the converging stage'.format(resume))
                        resume = 'converging_{}'.format(resume)
                    strict = False
                f = os.path.join(load, 'model', 'model_{}.pt'.format(resume))
        else:
            f = os.path.join(pretrain, 'model/model_latest.pt') if pretrain.find('.pt') < 0 else pretrain
            print('Load pre-trained model from {}'.format(f))
            strict = True

        if f:
            kwargs = {}
            if cpu:
                kwargs = {'map_location': lambda storage, loc: storage}
            state = torch.load(f, **kwargs)
            # for (k1, v1), (k2, v2) in zip(self.state_dict().items(), state.items()):
            #     print(k1, v1.shape)
            #     print(k2, v2.shape)
            self.get_model().load_state_dict(state, strict=strict)

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

