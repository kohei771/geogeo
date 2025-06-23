import pickle

with open('train_newmethod.pkl', 'rb') as f:
    data = pickle.load(f)

# 先頭5件だけ表示
for i, entry in enumerate(data[:5]):
    print(f"Sample {i}:")
    print("  pcd0:", entry['pcd0'])
    print("  pcd1:", entry['pcd1'])
    # 他のキーも見たい場合は print(entry.keys())