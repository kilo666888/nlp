
# 项目
## 代码核心功能说明
### 1. 获取邮件内容并切词处理 (get_words 函数)
```python
def get_words(filename):
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)  # 过滤无效字符
            line = cut(line)  # jieba分词
            line = filter(lambda word: len(word) > 1, line)  # 过滤长度为1的词
            words.extend(line)
    return words
```
### 2.构建词汇表并提取频率较高的词(get_top_words 函数)
```python
def get_top_words(top_num):
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    for filename in filename_list:
        all_words.append(get_words(filename))  # 遍历所有邮件生成词库
    freq = Counter(chain(*all_words))  # 统计词频
    return [i[0] for i in freq.most_common(top_num)]  # 返回前 top_num 个高频词
```
### 3. 构建邮件的词向量 (vector 生成部分)
```python
vector = []
for words in all_words:
    word_map = list(map(lambda word: words.count(word), top_words))  # 统计每个特征词的词频
    vector.append(word_map)
vector = np.array(vector)  # 转换为 NumPy 数组
```
### 4.构建分类模型 (MultinomialNB Naive Bayes 模型)
```python
model = MultinomialNB()  # 初始化多项式朴素贝叶斯模型
model.fit(vector, labels)  # 使用词频向量和标签进行训练
```
### 5. 对未知邮件进行分类 (predict 函数)
```python
def predict(filename):
    words = get_words(filename)  # 预处理新邮件
    current_vector = np.array(tuple(map(lambda word: words.count(word), top_words)))  # 生成词频向量
    result = model.predict(current_vector.reshape(1, -1))  # 预测结果
    return '垃圾邮件' if result == 1 else '普通邮件'
```
### 6. 使用 SMOTE 进行数据平衡（后续部分）
```python
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
```
## 高频词/TF-IDF两种特征模式的切换方法
### 1.高频词 (Word Frequency) 特征模式
 -高频词特征模式是指根据每个词在整个文本中的出现频率来表示文本。该方法简单直观，生成的特征向量通常是每个词在文本中出现的次数。
### 2. TF-IDF 特征模式
 -TF-IDF 是一种考虑到词频和逆文档频率的加权方法，它不仅反映了一个词在文档中的重要性，还考虑了词在整个语料库中的分布情况。
### 3.如何切换
 -可以通过调整文本特征提取的方式来在高频词和TF-IDF之间进行切换。通常使用 sklearn 提供的 CountVectorizer 和 TfidfVectorizer 来处理文本特征。
 ```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# 假设我们有一个文本数据集
documents = [
    "This is a spam message",
    "This is a ham message",
    "Free money is waiting for you",
    "Win a lottery by sending your email",
]
# 高频词特征模式：使用 CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
X_count = count_vectorizer.fit_transform(documents)
print("High-frequency features:")
print(count_vectorizer.get_feature_names_out())  # 输出特征词汇
# TF-IDF 特征模式：使用 TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(documents)
print("\nTF-IDF features:")
print(tfidf_vectorizer.get_feature_names_out())  # 输出特征词汇
```
## 运行结果
![屏幕截图 2025-04-09 193121](https://github.com/Luuu13/natural-language-processing/blob/master/image/test%204-1.png)
## 样本平衡处理
![屏幕截图 2025-04-09 193121](https://github.com/Luuu13/natural-language-processing/blob/master/image/test%204-2.png)
## 增加模型评估指标
![屏幕截图 2025-04-09 193121](https://github.com/Luuu13/natural-language-processing/blob/master/image/test%204-3.png)
