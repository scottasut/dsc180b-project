import sys
import logging

logging.basicConfig(filename='log.txt', 
    filemode='a',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    format='%(asctime)s %(levelname)s %(message)s')

sys.path.insert(0, 'src')

from dataset.make_dataset import process_data
from features.build_features import generate_features

def main(targets):
    if 'data' in targets:
        process_data(2010, 12)
    if 'features' in targets:
        generate_features()
    if 'test' in targets:
        pass

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
