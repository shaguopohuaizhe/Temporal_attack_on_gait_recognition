from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation
from config import conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


m = initialization(conf, test=opt.cache)[0]

# load model checkpoint of iteration opt.iter
print('Loading the model of iteration %d...' % opt.iter)
m.load(opt.iter)
print('Transforming...')
time = datetime.now()
m.evaluate('test', opt.batch_size)
