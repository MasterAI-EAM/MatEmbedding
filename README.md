# MAT_prediction
Main idea: 
- Get embeddings:
  - mat2vec (from Berkeley, pretrained on material paper abstracts)
  - word2vec trained by ourselves: specific domain embedding, using full-text papers
  - BERT embeddings
  - MATBERT embeddings
  - EnergyBERT embeddings (pretrained by us)
  - GPT embeddings (through openAI)
  - LLaMA2 embeddings
  - DARWIN embeddings (pretrained by us)
- calculate similarity of material names (filtered by NER tools) and selected word (for examples, "thermoelectric")
  - need to recognize material names & other useful phrases first before training embeddings
  - **Is there any ways to improve tokenizer for domain-specific LLM? need pre or post process?**
- eval: use experimental data (score of material names) to rank and compare correlation.
- for prediction show: divide top material names into used and unused. Unused ones are potential good ones.
