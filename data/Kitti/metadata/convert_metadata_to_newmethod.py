import os
import os.path as osp
import pickle

def update_metadata_pkl(in_pkl, out_pkl, old_key, new_key):
    with open(in_pkl, 'rb') as f:
        data = pickle.load(f)
    for entry in data:
        for k in ['pcd0', 'pcd1']:
            if entry[k].startswith(old_key):
                entry[k] = entry[k].replace(old_key, new_key, 1)
    with open(out_pkl, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    meta_dir = '.'
    old_key = 'downsampled/'
    new_key = 'newmethod/'
    for split in ['train', 'val', 'test']:
        in_pkl = osp.join(meta_dir, f'{split}.pkl')
        out_pkl = osp.join(meta_dir, f'{split}_newmethod.pkl')
        update_metadata_pkl(in_pkl, out_pkl, old_key, new_key)
    print('done')
