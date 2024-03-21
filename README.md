# MAT_prediction
## Background:

Since word2vec is static and dynamic embeddings (like BERT) have more advantages (no OOV problem, token representation, contextual representation), we want to see the performance of BERT embeddings on material prediction.

What we did in the paper "Tokenizer Effect on Functional Material Prediction: Investigating Contextual Word Embeddings for Knowledge Discovery": we compare three kinds of embedding: BERT, MatBERT, OpenAI embeddings from different layers of the model.

Methods: 1) context-free: no context, average of token embeddings, for example, SiO2 embedding = avg("Si#" embedding + "#O2" embedding).\n
dhdhdh



