import pickle
import os

with open('train_newmethod.pkl', 'rb') as f:
    data = pickle.load(f)

missing = []
for i, entry in enumerate(data):
    for k in ['pcd0', 'pcd1']:
        path = os.path.join('../newmethod', entry[k][9:]) if entry[k].startswith('newmethod/') else entry[k]
        # 絶対パスで確認したい場合は適宜修正
        if not os.path.exists(path):
            missing.append((i, k, entry[k]))

if missing:
    print(f"Missing files ({len(missing)}):")
    for i, k, p in missing[:20]:
        print(f"Sample {i} {k}: {p}")
    if len(missing) > 20:
        print(f"...and {len(missing)-20} more.")
else:
    print("All files exist.")
