"""更新 homework_train.py 适配新数据集"""
code = open('homework_train.py', 'r', encoding='utf-8').read()

# 1. 数据路径
code = code.replace(
    'D:/qdmsa_data/mosei/processed.pkl',
    'D:/qdmsa_data/mosei_full/processed_full.pkl')

# 2. 数据量
code = code.replace(
    'n_train, n_val, n_test = 2400, 400, 400',
    'n_train, n_val, n_test = 16000, 3000, 3856')

# 3. 7分类 → 3分类
code = code.replace('7  # 7分类输出', '3  # 3分类输出')
code = code.replace('output_shape=7', 'output_shape=3')
code = code.replace('minlength=7', 'minlength=3')

# 4. 情感标签名
old_names = "['-3(极负)', '-2', '-1', '0(中性)', '+1', '+2', '+3(极正)']"
new_names = "['负向', '中性', '正向']"
code = code.replace(old_names, new_names)

# 5. 随机基线
code = code.replace("0.143", "0.333")
code = code.replace("14.3%", "33.3%")

# 6. 论文精度提示
code = code.replace("81.62% on full CMU-MOSEI", "~80%+ on full CMU-MOSEI")

open('homework_train.py', 'w', encoding='utf-8').write(code)
print('Updated homework_train.py successfully!')
print('  Data: 22,856 samples (16,000 train + 3,000 val + 3,856 test)')
print('  Classes: 3 (负向/中性/正向)')
print('  Path: D:/qdmsa_data/mosei_full/processed_full.pkl')
