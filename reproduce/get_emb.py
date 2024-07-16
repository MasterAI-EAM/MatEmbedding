from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import json


# word embedding
w2v_model = Word2Vec.load("mat2vec/training/models/pretrained_embeddings")
word_vectors = w2v_model.wv
print(len(word_vectors.vocab))
centor = word_vectors['thermoelectric']
small = []
for word in word_vectors.vocab:
    c = word_vectors.vocab[word].count
    if c <= 3:
        small.append(word)
json_str = json.dumps(small, indent=4)
with open('smaller_3.json', 'w', encoding='utf-8') as json_file:
    json.dump(small, json_file)
word_vectors.save_word2vec_format('vectors.txt')
sims = word_vectors.most_similar('thermoelectric', topn=529686)
print(type(sims))
json_str = json.dumps(sims, indent=4)
with open('emb_sims.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_str)
'''
# output embedding
model = Word2Vec.load("mat2vec/training/models/pretrained_embeddings")
outv = KeyedVectors(529686)
outv.vocab = model.wv.vocab
outv.index2word = model.wv.index2word
outv.syn0 = model.syn1neg
sims = outv.most_similar(positive=[model.syn1neg[model.wv.vocab['thermoelectric'].index]], topn=529686)
json_str = json.dumps(sims, indent=4)
with open('emb_sims.json', 'w', encoding='utf-8') as json_file:
    json_file.write(json_str)
'''