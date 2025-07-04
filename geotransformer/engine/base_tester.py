import sys
import argparse
import os.path as osp
import time
import json
import abc

import torch
import ipdb

from geotransformer.utils.torch import initialize
from geotransformer.engine.logger import Logger


def inject_default_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default=None, help='load from snapshot')
    parser.add_argument('--test_epoch', type=int, default=None, help='test epoch')
    parser.add_argument('--test_iter', type=int, default=None, help='test iteration')
    return parser


class BaseTester(abc.ABC):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        # parser
        parser = inject_default_parser(parser)
        self.args = parser.parse_args()

        # logger
        log_file = osp.join(cfg.log_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
        self.logger = Logger(log_file=log_file)

        # command executed
        message = 'Command executed: ' + ' '.join(sys.argv)
        self.logger.info(message)

        # find snapshot
        if self.args.snapshot is None:
            if self.args.test_epoch is not None:
                self.args.snapshot = osp.join(cfg.snapshot_dir, 'epoch-{}.pth.tar'.format(self.args.test_epoch))
            elif self.args.test_iter is not None:
                self.args.snapshot = osp.join(cfg.snapshot_dir, 'iter-{}.pth.tar'.format(self.args.test_iter))
        if self.args.snapshot is None:
            raise RuntimeError('Snapshot is not specified.')

        # print config
        message = 'Configs:\n' + json.dumps(cfg, indent=4)
        self.logger.info(message)

        # cuda and distributed
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA devices available.')
        self.cudnn_deterministic = cudnn_deterministic
        self.seed = cfg.seed
        initialize(seed=self.seed, cudnn_deterministic=self.cudnn_deterministic)

        # state
        self.model = None
        self.iteration = None

        self.test_loader = None
        self.saved_states = {}

    def load_snapshot(self, snapshot):
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
        assert 'model' in state_dict, 'No model can be loaded.'
        # 手法名でstrict分岐
        exp_name = getattr(self, 'exp_name', None)
        if exp_name is None and hasattr(self, 'args') and hasattr(self.args, 'exp_name'):
            exp_name = self.args.exp_name
        if exp_name is None and hasattr(self, 'cfg') and hasattr(self.cfg, 'exp_name'):
            exp_name = self.cfg.exp_name
        # cfgがあればそちらを優先
        if hasattr(self, 'cfg') and hasattr(self.cfg, 'exp_name'):
            exp_name = self.cfg.exp_name
        # intensitygrad系ならstrict=False（新しい重み構造用）、それ以外はstrict=True（旧手法の重み構造用）
        if exp_name is not None and 'intensitygrad' in exp_name:
            strict_flag = False
        else:
            strict_flag = True
        self.model.load_state_dict(state_dict['model'], strict=strict_flag)
        self.logger.info(f'Model has been loaded. (strict={strict_flag})')

    def register_model(self, model):
        r"""Register model. DDP is automatically used."""
        self.model = model
        message = 'Model description:\n' + str(model)
        self.logger.info(message)
        return model

    def register_loader(self, test_loader):
        r"""Register data loader."""
        self.test_loader = test_loader

    @abc.abstractmethod
    def run(self):
        raise NotImplemented
