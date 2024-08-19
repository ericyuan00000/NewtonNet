#! /usr/bin/env python

import os
import argparse
import yaml

import torch
from newtonnet.data import parse_train_test
from newtonnet.layers.precision import get_precison_by_string
# torch.autograd.set_detect_anomaly(True)

# argument parser description
parser = argparse.ArgumentParser(
    description='This is a pacakge to train NewtonNet on a given data.',
    )
parser.add_argument(
    '-c',
    '--config',
    type=str,
    help='The path to the Yaml configuration file.',
    )

# define arguments
args = parser.parse_args()
config = args.config

# locate files
settings_path = os.path.abspath(config)
settings = yaml.safe_load(open(settings_path, 'r'))

# device
precision = get_precison_by_string(settings['general']['precision'])

# data
torch.manual_seed(settings['general']['seed'])
settings['data'].pop('cutoff')
train_gen, val_gen, test_gen, stats = parse_train_test(
    precision=precision,
    **settings['data'],
    )