"""
将 tamb2203579/CMU-MOSEI 的 aligned_50.pkl 转换为 homework_train.py 可用格式
"""
import pickle, numpy as np

print("Loading aligned_50.pkl...")
with open('D:/qdmsa_data/mosei_full/Processed/aligned_50.pkl', 'rb') as f:
    src = pickle.load(f)

all_samples = []

for split_name in ['train', 'valid', 'test']:
    d = src[split_name]
    n = d['audio'].shape[0]  # 段数
    labels = d['classification_labels']  # (N, 7) one-hot
    # 转成单标签: argmax → 0~6
    if labels.ndim == 2:
        labels_1d = np.argmax(labels, axis=1)
    else:
        labels_1d = labels

    for i in range(n):
        # 取每个段的三模态数据
        audio_i = d['audio'][i].astype(np.float32)     # (50, 74)
        vision_i = d['vision'][i].astype(np.float32)   # (50, 35)

        # text 可能是 GloVe (N, 50, 300) 或原始文本
        if hasattr(d['text'], 'shape') and d['text'].ndim == 3:
            text_i = d['text'][i].astype(np.float32)   # (50, 300)
        else:
            # 没有 GloVe，用零占位
            text_i = np.zeros((50, 300), dtype=np.float32)

        all_samples.append({
            'text': text_i,
            'audio': audio_i,
            'vision': vision_i,
            'label': int(labels_1d[i]),
        })

print(f"Total samples: {len(all_samples)}")

# 保存
out_path = 'D:/qdmsa_data/mosei_full/processed_full.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(all_samples, f)

import os
size = os.path.getsize(out_path) / 1024 / 1024
print(f"Saved to: {out_path} ({size:.0f} MB)")

# 标签分布
from collections import Counter
lc = Counter(s['label'] for s in all_samples)
print(f"Label distribution: {dict(sorted(lc.items()))}")
