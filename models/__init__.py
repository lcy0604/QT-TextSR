# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .pix2seq_sr import build


def build_model(args):
    return build(args)
