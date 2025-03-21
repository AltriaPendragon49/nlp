# 三元语法模型（Trigram Model）

这个项目实现了一个基于NLTK的三元语法模型，用于文本预测和熵分析。

## 功能特点

1. 文本预处理和模型训练
   - 使用NLTK的Reuters语料库
   - 实现三元语法模型的训练
   - 计算词序列的条件概率

2. 词语预测
   - 基于前两个词预测下一个最可能出现的词
   - 使用概率最大化方法进行预测

3. 熵分析
   - 计算句子中每个词的信息熵
   - 可视化展示熵值分布

## 环境要求

- Python 3.x
- NLTK
- matplotlib

## 安装依赖

```
pip install nltk matplotlib
python -c "import nltk; nltk.download('reuters'); nltk.download('punkt')"
```

## 使用方法

### 1. 预测下一个词

```python
from trigram import predict_next_word

# 预测"the economy"后最可能出现的词
prediction = predict_next_word("the", "economy")
print(f"预测的下一个词: {prediction}")
```

### 2. 计算句子熵值

```python
from trigram import calculate_entropy

# 计算句子中每个词的熵值
sentence = "The economy is growing steadily"
entropy_values = calculate_entropy(sentence)
```

### 3. 可视化熵值分布

```python
import matplotlib.pyplot as plt

plt.plot(entropy_values)
plt.xlabel('句子中的词位置')
plt.ylabel('熵')
plt.title('句子中词的熵值')
plt.show()
```

## 技术说明

1. 模型训练过程
   - 使用Reuters语料库的文本数据
   - 创建词序列的三元组
   - 统计三元组出现频率
   - 计算条件概率

2. 预测算法
   - 基于条件概率选择最可能的下一个词
   - 当没有匹配的三元组时返回"无可用预测"

3. 熵计算
   - 使用信息论中的熵公式计算
   - 反映词序列的不确定性

## 注意事项

1. 首次运行需要下载NLTK资源
2. 模型训练可能需要较长时间
3. 预测结果依赖于训练数据的质量和覆盖范围

## 改进方向

1. 添加数据预处理步骤，提高模型质量
2. 实现模型持久化，避免重复训练
3. 添加更多评估指标
4. 优化内存使用
5. 支持自定义训练语料库