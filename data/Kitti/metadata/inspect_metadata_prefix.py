import os.path as osp
import pickle

def print_unique_prefixes(pkl_path, key='pcd0'):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    prefixes = set()
    for entry in data:
        for k in ['pcd0', 'pcd1']:
            path = entry[k]
            # 先頭3階層くらいまで
            parts = path.split('/')
            prefix = '/'.join(parts[:3])
            prefixes.add(prefix)
    print(f"{pkl_path} unique prefixes:")
    for p in sorted(prefixes):
        print(p)

if __name__ == '__main__':
    for split in ['train', 'val', 'test']:
        print_unique_prefixes(f'{split}.pkl')
