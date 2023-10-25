import yaml
import easydict
from os.path import join


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


import argparse
parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

args = parser.parse_args()

config_file = args.config

args = yaml.safe_load(open(config_file))

save_config = yaml.safe_load(open(config_file))

args = easydict.EasyDict(args)

dataset = None

if args.data.dataset.name == 'officehome':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['Target', 'Source', 'Source2', 'Source3'],
    files=[
        'Source_train.csv',
        'Target_train.csv',
        'Source_train.csv',
        'Source_train.csv'
    ],
    prefix=args.data.dataset.root_path)

else:
    raise Exception(f'dataset {args.data.dataset.name} not supported!')

source1_domain_name = dataset.domains[args.data.dataset.source1]
target_domain_name = dataset.domains[args.data.dataset.target]
source1_file = dataset.files[args.data.dataset.source1]
target_file = dataset.files[args.data.dataset.target]
if args.data.dataset.source2 != None:
    source2_domain_name = dataset.domains[args.data.dataset.source2]
    source2_file = dataset.files[args.data.dataset.source2]
if args.data.dataset.source3 != None:
    source3_domain_name = dataset.domains[args.data.dataset.source3]
    source3_file = dataset.files[args.data.dataset.source3]
if args.data.dataset.source4 != None:
    source4_domain_name = dataset.domains[args.data.dataset.source4]
    source4_file = dataset.files[args.data.dataset.source4]
