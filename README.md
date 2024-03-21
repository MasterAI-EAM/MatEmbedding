# MAT prediction version 1.0
## Background:
Since word2vec is static and dynamic embeddings (like BERT) have more advantages (no OOV problem, token representation, contextual representation), we want to see the performance of BERT embeddings on material prediction.

What we did in the paper "Tokenizer Effect on Functional Material Prediction: Investigating Contextual Word Embeddings for Knowledge Discovery": We compare three kinds of embedding: BERT, MatBERT, OpenAI embeddings from different layers of the model.

Methods: 
- context-free: no context, average of token embeddings, for example, SiO2 embedding = avg("Si#" embedding + "#O2" embedding).
- context-average: the material names are in the sentence. Use the average of contextual token embedding (or `<CLS>` token).

Note: Since OpenAI embedding cannot generate contextual token embedding, we just use the embedding of material names.

The results show that context-averaged BERT embeddings are better than context-free ones, but still not as good as Word2Vec embedding in the thermoelectric material prediction task.

We further improve MatBERT embedding by contrastive learning and add more experiments to systematically evaluate the quality of embedding & performance in downstream tasks.

## Datasets for training:
- [ ] Word2vec material names (pairs with similarity scores) - support supervised training [prepared by Yuwei]
- [ ] title and abstracts - support supervised training & unsupervised training [prepared by Yuwei] 
- [x] Wiki material formula, names, and description - support supervised & unsupervised training [prepared by Nan]

## Metrics/Datasets for evaluation:
- [ ] Anistropy (a relative value. Use sentence corpus, for example, description list, or material name list) 
- [x] 100 material names (50 similar, 50 not similar)
- [x] thermoelectric zT, use rank correlation
- [ ] other kind of property (more data for training regression model)

## Experiments 1.0:
train MatBERT with Wiki data, compare performance before & after (including different layers, different methods, and different tasks)








