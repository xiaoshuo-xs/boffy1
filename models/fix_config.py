"""一键修复 homework_train.py 为 cairocode 3分类 + 无TT压缩"""
code = open('homework_train.py', 'r', encoding='utf-8').read()

# 数据路径和量
code = code.replace('D:/qdmsa_data/mosei_star/processed_star.pkl',
                    'D:/qdmsa_data/mosei_full/processed_full.pkl')
code = code.replace('n_train, n_val, n_test = 16326, 1871, 4659',
                    'n_train, n_val, n_test = 16000, 3000, 3856')

# 3分类
code = code.replace('minlength=4', 'minlength=3')
code = code.replace('minlength=5', 'minlength=3')
code = code.replace('minlength=7', 'minlength=3')
code = code.replace('output_shape=4', 'output_shape=3')
code = code.replace('output_shape=5', 'output_shape=3')
code = code.replace('output_shape=7', 'output_shape=3')
code = code.replace('nn.Linear(128, 4)', 'nn.Linear(128, 3)')
code = code.replace('nn.Linear(128, 5)', 'nn.Linear(128, 3)')
code = code.replace('nn.Linear(128, 7)', 'nn.Linear(128, 3)')
code = code.replace('weights.sum() * 4', 'weights.sum() * 3')
code = code.replace('weights.sum() * 5', 'weights.sum() * 3')
code = code.replace('weights.sum() * 7', 'weights.sum() * 3')

# 标签名
for old in ["['label-0','label-1','label-2','label-3']",
            "['-2','-1','0','+1','+2']",
            "['-3','-2','-1','0','+1','+2','+3']"]:
    code = code.replace(old, "['负面','中性','正面']")

# 基线
code = code.replace('0.25', '0.333')
code = code.replace('25%', '33.3%')
code = code.replace('0.143', '0.333')
code = code.replace('14.3%', '33.3%')

# ★ 去掉 TT 压缩
code = code.replace('tt_rank=8', 'tt_rank=0')

# 注释
code = code.replace('4  # 4类情感标签', '3  # 3类情感标签')
code = code.replace('5  # 5类情感标签', '3  # 3类情感标签')
code = code.replace('7  # 7类情感标签', '3  # 3类情感标签')
code = code.replace('3  # 3分类输出', '3  # 3分类输出')
code = code.replace('3分类随机猜', '3分类随机猜')

open('homework_train.py', 'w', encoding='utf-8').write(code)

# 验证
for line in open('homework_train.py', 'r', encoding='utf-8'):
    for kw in ['n_train', 'output_shape', 'processed', 'minlength', 'tt_rank', 'weights.sum']:
        if kw in line:
            print(f'  {line.strip()}')
    if 'emotion_names' in line:
        print(f'  {line.strip()}')
print('Done!')
