## Stage1
1. Data Sampling (the_verdict.txt is used) and Preparation techniques
2. Implementing each component of transformer
    - Tokenizer : Various methods of doing tokenization (from scratch and also 
                  from tiktoken library) 
    - Positional Encodings 
    - Self Attention / Causal Attention / Multihead Self Attention
    - Layer Normalization
    - GeLU activation
    - Feed Forward Network
3. Building GPT-2 architecture 

## Stage2
1. Pretraining LLM for making foundational Model
    - Blocks of LLM
    - Added training loop 
    - Added evaluation loop 
    - Save and load model weights
    - Loading OpenAI GPT2 weights 

## Stage3 
- Finetuning for Classification of Emails as `spam` and `ham`
