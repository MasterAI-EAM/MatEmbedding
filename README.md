# MAT_prediction
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
- word2vec material names (pairs with similarity scores) [prepared by Yuwei]
- 






