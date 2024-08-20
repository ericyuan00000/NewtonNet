#! /usr/bin/env python

import argparse

from newtonnet.data import RadiusGraph
from newtonnet.data import parse_train_test
from newtonnet.layers.precision import get_precison_by_string
# torch.autograd.set_detect_anomaly(True)

# argument parser description
parser = argparse.ArgumentParser(
    description='This is a pacakge to train NewtonNet on a given data.',
    )
parser.add_argument(
    '-r',
    '--root',
    type=str,
    help='The path to the raw data root directory.',
    )
parser.add_argument(
    '-p',
    '--precision',
    type=str,
    help='The precision of the model. Default: single.',
    default='single',
)

# define arguments
args = parser.parse_args()
root = args.root
precision = args.precision

# device
precision = get_precison_by_string(precision)

# data
MolecularDataset(root=root, precision=precision, force_reload=True)

print('done!')