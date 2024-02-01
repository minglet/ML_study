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
    
def create_co_matrix():

    return 0

def cos_similarity():
    return 0
