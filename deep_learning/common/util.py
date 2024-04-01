import sys
sys.path.append('..')
import os
from common.np import *

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id

    id_to_word = {id_: word for word, id_ in word_to_id.items()} # id가 key값으로 들어감
    corpus = np.array([word_to_id[word] for word in words])
    return corpus, word_to_id, id_to_word
    
def create_contexts_target(corpus, window_size=1):
    '''
    context, target 생성
    corpus: 말뭉치 (단어 ID 목록)
    window_size: 윈도우 크기
    return (context, target)의 array
    '''

    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus)-window_size): 
        cs = []
        # window 크기만큼 타겟 단어 좌우 context 가져오기 
        # [-1, 0, 1] 이렇게 배치되어있을 때 0은 target, -1, 1은 contexts를 의미함
        for t in range(-window_size, window_size+1):
            if t != 0:
                cs.append(corpus[idx + t]) # context인 경우 cs에 추가 / idx의 의미는 해당 corpus[idx]가 target이라는 말
        contexts.append(cs)
    
    return np.array(contexts), np.array(target)

def convert_one_hot(corpus, vocab_size):
    '''
    원핫 표현으로 변환
    corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    vocab_size: 어휘 수
    return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    '''
    N = corpus.shape[0]
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate

def cos_similarity():
    return 0
