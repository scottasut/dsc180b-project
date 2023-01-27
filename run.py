import sys

sys.path.insert(0, 'src')

from data.create_network import build_graph

def main(targets):
    if 'data' in targets:
        build_graph()

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
