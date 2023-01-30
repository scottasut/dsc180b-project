import sys

sys.path.insert(0, 'src')

from dataset.create_dataset import build_graph

def main(targets):
    if 'data' in targets:
        build_graph()

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
