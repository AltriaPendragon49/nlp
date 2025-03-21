# coding=utf-8
import nltk
from nltk.corpus import reuters
from nltk import trigrams
from collections import defaultdict
import pdb
import math
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 下载必要的NLTK资源
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("正在下载punkt_tab资源...")
    nltk.download('punkt_tab')

nltk.download('reuters')
nltk.download('punkt')

# 对文本进行分词
words = nltk.word_tokenize(' '.join(reuters.words()))

# 创建三元语法
tri_grams = list(trigrams(words))

# 构建三元语法模型
model = defaultdict(lambda: defaultdict(lambda: 0))

# 统计共现频率
for w1, w2, w3 in tri_grams:
    model[(w1, w2)][w3] += 1

# 将计数转换为概率
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

def predict_next_word(w1, w2):
    """
    基于前两个词使用训练好的三元语法模型预测下一个词。
    参数:
    w1 (str): 第一个词
    w2 (str): 第二个词
    返回:
    str: 预测的下一个词
    """
    next_word = model[w1, w2]
    
    # 使用pdb检查模型和概率（用于调试）
    #pdb.set_trace()
    
    if next_word:
        predicted_word = max(next_word, key=next_word.get)  # 选择最可能的下一个词
        return predicted_word
    else:
        return "无可用预测"

def calculate_entropy(sentence):
    """
    计算句子中每个词的信息熵。
    参数:
    sentence (str): 一个句子
    返回:
    list: 每个词的熵值列表
    """
    words_in_sentence = nltk.word_tokenize(sentence)
    entropy_values = []
    
    for i in range(len(words_in_sentence) - 2):
        w1, w2 = words_in_sentence[i], words_in_sentence[i + 1]
        
        # 获取下一个词的概率分布
        next_word_dist = model.get((w1, w2), {})
        
        if next_word_dist:
            total_count = float(sum(next_word_dist.values()))
            entropy = 0
            for word, prob in next_word_dist.items():
                entropy -= prob * math.log2(prob)
            entropy_values.append(entropy)
        else:
            # 如果没有可用预测，假设高熵值
            entropy_values.append(1.0)
    
    return entropy_values

# 测试预测和熵计算
prediction = predict_next_word("normal", "humidity")
print(f"预测的下一个词: {prediction}")

# 用于熵计算的示例句子
sentence = "The economy is growing steadily"

# 计算句子中每个词的熵
entropy_values = calculate_entropy(sentence)

# 绘制熵值图
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(entropy_values, marker='o')  # 添加数据点标记
plt.xlabel('句子中的词位置', fontsize=12)
plt.ylabel('熵', fontsize=12)
plt.title('句子中词的熵值分布', fontsize=14)
plt.grid(True)  # 添加网格线
plt.show()
