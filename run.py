import sys
import logging

logging.basicConfig(filename='log.txt', 
    filemode='a', 
    level=logging.INFO,
    datefmt='%H:%M:%S',
    format='%(asctime)s %(levelname)s %(message)s')

sys.path.insert(0, 'src')

from dataset.create_dataset import build_graph, build_test_set

def main(targets):
    if 'data' in targets:
        build_graph()
        build_test_set()
    if 'test' in targets:
        pass

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
