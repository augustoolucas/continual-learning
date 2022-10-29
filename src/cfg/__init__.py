import yaml

with open('./cfg/config.yaml', 'r') as reader:
    cfg = yaml.load(reader, Loader=yaml.SafeLoader)

train_cfg = cfg['train']
test_cfg = cfg['test']
