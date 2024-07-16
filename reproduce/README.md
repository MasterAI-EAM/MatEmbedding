# File description
- get_emb.py is a load code of mat2vec embedding
- reproduce.ipynb is code that tries to reproduce results of mat2vec paper
- prediction.ipynb is an example of using mat2vec to predict rank
- matscholar.py is a file from mat2vec repo because you need to use the specific format they use (for example, LiCoO2 will be CoLiO2 in their embeddings). you need to download [phraser.pkl](https://github.com/materialsintelligence/mat2vec/tree/master/mat2vec/processing/models) and put it in the same folder.
- PF.txt & zt.json is the data files
# Usage note
input files you need to run reproduce.ipynb: embedding_sims.json / output_sims (by running get_emb.py)


