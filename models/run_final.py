"""最终配置：starhu123 GloVe 300 + 3分类 + tt_rank=0"""
code = open('homework_train.py', 'r', encoding='utf-8').read()

# 数据
code = code.replace('D:/qdmsa_data/mosei_full/processed_full.pkl',
                    'D:/qdmsa_data/mosei_star/processed_v2.pkl')
code = code.replace('D:/qdmsa_data/mosei_star/processed_star.pkl',
                    'D:/qdmsa_data/mosei_star/processed_v2.pkl')

# 数据量 (starhu123 分好的: 16326/1871/4659)
code = code.replace('n_train, n_val, n_test = 16000, 3000, 3856',
                    'n_train, n_val, n_test = 16326, 1871, 4659')

# 3分类
for old, new in [('minlength=7','minlength=3'), ('minlength=4','minlength=3'),
                 ('minlength=5','minlength=3')]:
    code = code.replace(old, new)

for old, new in [('output_shape=7','output_shape=3'), ('output_shape=4','output_shape=3'),
                 ('output_shape=5','output_shape=3')]:
    code = code.replace(old, new)

code = code.replace('nn.Linear(128, 7)', 'nn.Linear(128, 3)')
code = code.replace('nn.Linear(128, 4)', 'nn.Linear(128, 3)')
code = code.replace('nn.Linear(128, 5)', 'nn.Linear(128, 3)')

for old, new in [('weights.sum() * 7','weights.sum() * 3'),
                 ('weights.sum() * 4','weights.sum() * 3'),
                 ('weights.sum() * 5','weights.sum() * 3')]:
    code = code.replace(old, new)

# 标签名
for old in ["['-3','-2','-1','0','+1','+2','+3']",
            "['label-0','label-1','label-2','label-3']",
            "['-2','-1','0','+1','+2']"]:
    code = code.replace(old, "['负面','中性','正面']")

# 随机基线
for old, new in [('0.25','0.333'), ('0.143','0.333'), ('25%','33.3%'), ('14.3%','33.3%')]:
    code = code.replace(old, new)

# TT压缩
code = code.replace('tt_rank=8', 'tt_rank=0')
code = code.replace('tt_rank=4', 'tt_rank=0')

# 注释统一
for label_line in ['7  # 7分类输出','4  # 4分类输出','5  # 5分类输出',
                   '7  # 7类情感标签','4  # 4类情感标签','5  # 5类情感标签']:
    code = code.replace(label_line, '3  # 3分类输出')

open('homework_train.py', 'w', encoding='utf-8').write(code)
print('Config updated: starhu123 GloVe 300, 3-class, tt_rank=0')
