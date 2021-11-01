import importlib
import Frameworks
from Frameworks import Framework
importlib.reload(Frameworks)
import pickle 
import copy
from paras import samples 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--if_text', type=bool, default=False, help='')
parser.add_argument('--if_linking', type=bool, default=False, help='')
parser.add_argument('--if_true_label', type=bool, default=False, help='')
parser.add_argument('--if_interact', type=bool, default=False, help='')
parser.add_argument('--if_infer', type=bool, default=True, help='')
parser.add_argument('--if_adjust', type=bool, default=False, help='')
parser.add_argument('--sampling_type', type=str, default='random', help='')
parser.add_argument('--select_type', type=str, default='random', help='')
args = parser.parse_args()

# if_text=True, if_true_label=True, if_linking=True, if_interact=False, if_infer=True, if_adjust=False, sampling_type='wstratified', select_type='MCTS', samples=samples

# 分层抽样不调整
framework = Framework(if_text=args.if_text, if_true_label=args.if_true_label, if_linking=args.if_linking, if_interact=args.if_interact, if_infer=args.if_infer, if_adjust=args.if_adjust, sampling_type=args.sampling_type, select_type=args.select_type, samples=samples)

framework.run()
