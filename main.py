import gensim
import numpy as np

# PATHは学習済みファイルの保存場所
PATH = "src/latest-ja-word2vec-gensim-model/word2vec.gensim.model"
model = gensim.models.Word2Vec.load(PATH)

q = model.wv["数学"]
d = model.wv["物理"]

# 類似度
def cal_cos_similarity(q, d):
    return np.dot(q, d) / np.linalg.norm(q) * np.linalg.norm(d)

similarity = cal_cos_similarity(q, d)

print(similarity)
# q = input('検索ワードを入力してください: ')