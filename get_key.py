from pyhanlp import *
from django.http import HttpResponse, HttpRequest, JsonResponse, HttpResponseBadRequest
import pandas as pd
import jieba
import jieba.analyse
import numpy as np
import gensim
from collections import Counter

# 测试时，可以注释掉下面一行，第四个功能作废
model = gensim.models.word2vec.Word2Vec.load('get_key/baike_26g_news_13g_novel_229g.model')


def get_key_1(request):
    # return HttpResponse('成功')
    payload = request.POST
    content = payload['parm']
    TextRankKeyword = JClass("com.hankcs.hanlp.summary.TextRankKeyword")
    keyword_list = HanLP.extractKeyword(content, 5)
    return HttpResponse('关键词为：' + '、'.join(keyword_list) + '<br>（默认五个关键词）')


def get_key_2(request):
    payload = request.POST
    content = payload['parm']
    stopwords = pd.read_csv('get_key/stopwords.txt', sep='\t', quoting=3, names=['stopwords'], index_col=False,
                            encoding='utf-8')
    # print(stopwords.head())

    # 将DateFrame的stopwords数据转换为list形式
    clean_content, all_words = drop_stops(content, stopwords)
    # print(clean_content)

    content_word = ''.join(clean_content)

    content_text = '、'.join(jieba.analyse.extract_tags(content_word, topK=5, withWeight=False))
    # print(content_word)
    # print(content_text)

    return HttpResponse('关键词为：' + content_text + '<br>（默认五个关键词）<br>（效果不好）')


def get_key_3(request):
    # return HttpResponse('成功3')
    payload = request.POST
    content = payload['parm']
    stopwords = pd.read_csv('get_key/stopwords.txt', sep='\t', quoting=3, names=['stopwords'], index_col=False,
                            encoding='utf-8')
    # print(stopwords.head())

    # 将DateFrame的stopwords数据转换为list形式
    clean_content, all_words = drop_stops(content, stopwords)
    # print(clean_content)

    content_word = ''.join(clean_content)

    content_text = '、'.join(jieba.analyse.textrank(content_word, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')))
    # print(content_word)
    # print(content_text)

    return HttpResponse('关键词为：' + content_text + '<br>（最多20个关键词）<br>（效果不好）')


def get_key_4(request):
    # return HttpResponse('成功4')

    # model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')
    # model = gensim.models.word2vec.load('data.model')
    payload = request.POST
    content = payload['parm']
    result = pd.Series(keywords(jieba.cut(content)))
    # print(result)

    return HttpResponse('关键词为：<br>' + str(result).replace('\n', '<br>').replace('dtype: object', ''))


# 对文本进行停止词的去除
def drop_stops(Jie_content, stopwords):
    clean_content = []
    all_words = []
    #     for j_content in Jie_content:
    line_clean = []
    for line in Jie_content:
        if line in stopwords:
            continue
        line_clean.append(line)
        all_words.append(line)
    clean_content.append(line_clean)

    return clean_content[0], all_words


def predict_proba(oword, iword):
    iword_vec = model[iword]
    oword = model.wv.vocab[oword]
    oword_l = model.syn1[oword.point].T
    dot = np.dot(iword_vec, oword_l)
    lprob = -sum(np.logaddexp(0, -dot) + oword.code * dot)
    return lprob


def keywords(s):
    s = [w for w in s if w in model]
    ws = {w: sum([predict_proba(u, w) for u in s]) for w in s}
    return Counter(ws).most_common()
