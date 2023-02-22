import sys
import os
import json
import logging

logging.basicConfig(filename='log.txt', 
    filemode='a',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    format='%(asctime)s %(levelname)s %(message)s')

sys.path.insert(0, 'src')

from dataset.make_dataset import process_data
from features.build_features import generate_features

def load_config(path: str, required_params: list) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError('File \'{}\' is missing. Please refer to the README for project setup.'.format(path))

    with open(path) as f:
        config = json.load(f)
    
    for param in required_params:
        if param not in config:
            raise ValueError('One or more of the following arguments in \'{}\' are missing: {}'.format(path, required_params))
        
    return config

def main(targets: list):

    setup_config = load_config('config/setup.json', ['year', 'month', 'keyword_extractor'])

    if 'data' in targets:
        process_data(setup_config['year'], setup_config['month'])
        
    if 'features' in targets:
        try:
            generate_features(setup_config['keywords'], setup_config['keyword_extractor'])
        except FileNotFoundError:
            print('Unable to find data files for feature generation. Please try running \'data\' target first.')
            sys.exit(-1)
    
    if 'test' in targets:
        pass

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
