import os
import sys
import argparse
sys.path.insert(0, '../model')
import qnn
import torch

parser = argparse.ArgumentParser(description='kernel shape analysis')
parser.add_argument('--ref', type=str, default='../../models/resnet18.pt')
parser.add_argument('--n_bits', type=int, default=8)
args = parser.parse_args()

model_ref = torch.load(args.ref)
qnn.QuantizeParams.bits_w = args.n_bits

for k, v in model_ref.items():
    if k.find('scales') >= 0:
        model_ref[k] = qnn.QuantizeParams.apply(v)

path, ext = os.path.splitext(args.ref)
torch.save(model_ref, '{}_fix{}{}'.format(path, args.n_bits, ext))
