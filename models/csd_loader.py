"""
=============================================================================
 CSD 数据加载器 — 直接将 CMU-MOSEI 的 HDF5(.csd) 文件转为训练格式
=============================================================================
 背景: CMU-Multimodal SDK 的 align() 函数内存消耗极大（~14GB数据全载入内存），
       对大如 MOSEI 的数据集会 MemoryError。
       本脚本用 h5py 直接读取 CSD 文件，逐segment处理，内存友好。

 使用: python csd_loader.py mosei D:/qdmsa_data
=============================================================================
"""

import h5py
import numpy as np
import os
import pickle
from tqdm import tqdm


def load_csd_segments(csd_path):
    """
    读取单个 CSD 文件中的所有 segment
    CSD 文件结构:
      root → [modality_name] → data → [segment_id]
                                 → metadata
      每个 segment: features (N_timesteps, D) + intervals (N_timesteps, 2)

    返回: {seg_id: {"features": ndarray, "intervals": ndarray}}
    """
    f = h5py.File(csd_path, 'r')
    root_key = list(f.keys())[0]          # 如 'COVAREP' 或 'glove_vectors'
    data_group = f[root_key]['data']       # 所有segment的数据

    segments = {}
    for seg_id in data_group.keys():
        seg = data_group[seg_id]
        segments[seg_id] = {
            'features': seg['features'][:],    # 特征矩阵 (N, D)
            'intervals': seg['intervals'][:]    # 时间区间 (N, 2)
        }

    f.close()
    return segments


def load_labels(csd_path):
    """加载标签CSD — 和 load_csd_segments 结构相同"""
    return load_csd_segments(csd_path)


def process_dataset(text_csd, audio_csd, vision_csd, labels_csd,
                    output_path, dataset_type='mosei', max_segments=None):
    """
    处理4个CSD文件，生成统一格式的训练数据

    参数:
      text_csd:   文本特征 CSD 路径（GloVe词向量，300维）
      audio_csd:  音频特征 CSD 路径（COVAREP，74维）
      vision_csd: 视觉特征 CSD 路径（Facet，35维）
      labels_csd: 标签 CSD 路径（7类情感，one-hot）
      output_path:  输出 pickle 路径
      max_segments: 限制最大segment数（None=全部）

    返回: 预处理好的数据列表
    """
    # 加载4个CSD文件
    print("Loading text features (GloVe 300-dim)...")
    text_data = load_csd_segments(text_csd)
    print(f"  {len(text_data)} segments, dim={list(text_data.values())[0]['features'].shape[1]}")

    print("Loading audio features (COVAREP 74-dim)...")
    audio_data = load_csd_segments(audio_csd)
    print(f"  {len(audio_data)} segments, dim={list(audio_data.values())[0]['features'].shape[1]}")

    print("Loading vision features (Facet 35-dim)...")
    vision_data = load_csd_segments(vision_csd)
    print(f"  {len(vision_data)} segments, dim={list(vision_data.values())[0]['features'].shape[1]}")

    print("Loading labels (7-class sentiment)...")
    labels_data = load_labels(labels_csd)
    print(f"  {len(labels_data)} segments")

    # 找出4个文件中都有的segment
    common_ids = set(text_data.keys()) & set(audio_data.keys()) & \
                 set(vision_data.keys())
    common_ids = sorted(common_ids)
    print(f"\nCommon segments across all modalities: {len(common_ids)}")

    if max_segments:
        common_ids = common_ids[:max_segments]

    processed = []
    skipped = 0

    for seg_id in tqdm(common_ids, desc="Processing segments"):
        text_feat   = text_data[seg_id]['features']    # (T_text, 300)
        audio_feat  = audio_data[seg_id]['features']   # (T_audio, 74)
        vision_feat = vision_data[seg_id]['features']  # (T_vision, 35)

        # 提取segment级标签: 标签是多行one-hot（每行对应一个时间段），
        # 取所有行的 argmax 的众数作为整体情感标签
        if seg_id in labels_data:
            label_arr = labels_data[seg_id]['features']  # (N_labels, 7)
            if label_arr.shape[0] > 1:
                label_votes = np.argmax(label_arr, axis=1)
                seg_label = int(np.argmax(np.bincount(label_votes)))  # 众数
            else:
                seg_label = int(np.argmax(label_arr[0]))
        else:
            skipped += 1
            continue

        # 对齐时间步: 取三模态中最短的序列长度
        min_len = min(text_feat.shape[0], audio_feat.shape[0],
                      vision_feat.shape[0])
        if min_len < 2:
            skipped += 1
            continue

        processed.append({
            'text':   text_feat[:min_len].astype(np.float32),
            'audio':  audio_feat[:min_len].astype(np.float32),
            'vision': vision_feat[:min_len].astype(np.float32),
            'label':  seg_label,  # 0~6 对应 -3 ~ +3
            'seg_id': seg_id
        })

    print(f"\nProcessed: {len(processed)} segments, Skipped: {skipped}")
    if processed:
        print(f"  Text dim:   {processed[0]['text'].shape[1]}")
        print(f"  Audio dim:  {processed[0]['audio'].shape[1]}")
        print(f"  Vision dim: {processed[0]['vision'].shape[1]}")

    # 保存为 pickle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(processed, f)
    print(f"Saved to {output_path}")

    return processed


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python csd_loader.py [mosei|mosi] [output_dir]")
        sys.exit(1)

    dataset = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'D:/qdmsa_data'

    if dataset == 'mosei':
        data_dir = 'D:/qdmsa_data/mosei'
        process_dataset(
            text_csd=os.path.join(data_dir,
                'CMU_MOSEI_TimestampedWordVectors.csd'),
            audio_csd=os.path.join(data_dir,
                'CMU_MOSEI_COVAREP.csd'),
            vision_csd=os.path.join(data_dir,
                'CMU_MOSEI_VisualFacet42.csd'),
            labels_csd=os.path.join(data_dir,
                'CMU_MOSEI_Labels.csd'),
            output_path=os.path.join(output_dir, 'mosei', 'processed.pkl'),
            dataset_type='mosei',
            max_segments=None
        )
    else:
        print("MOSI not configured yet.")