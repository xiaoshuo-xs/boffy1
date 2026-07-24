code = open('homework_train.py', 'r', encoding='utf-8').read()

# 数据路径
code = code.replace(
    'D:/qdmsa_data/mosei_full/processed_full.pkl',
    'D:/qdmsa_data/mosei_star/processed_star.pkl')

# 数据量
code = code.replace(
    'n_train, n_val, n_test = 16000, 3000, 3856',
    'n_train, n_val, n_test = 16326, 1871, 4659')

# 4分类
code = code.replace('minlength=3', 'minlength=4')
code = code.replace('output_shape=3', 'output_shape=4')
code = code.replace('nn.Linear(128, 3)', 'nn.Linear(128, 4)')
code = code.replace("3  # 3类情感标签", "4  # 4类情感标签")
code = code.replace("weights.sum() * 3", "weights.sum() * 4")

# 标签名
old_names = "['负向', '中性', '正向']"
new_names = "['label-0','label-1','label-2','label-3']"
code = code.replace(old_names, new_names)

# 随机基线
code = code.replace("0.333", "0.25")
code = code.replace("33.3%", "25%")

open('homework_train.py', 'w', encoding='utf-8').write(code)
print('Updated for starhu123 dataset!')
print('  4 classes, 22856 samples, GloVe 300-dim')
