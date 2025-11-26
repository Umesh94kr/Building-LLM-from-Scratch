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
    - Finetuned GPT2 small by replacing output layer with classification and allowing last attention layer and final normalization layer to get trained (update parameters while training)

- Instruction Finetuning
    - Allows LLM to follow some particular instruction
    - Finetuned all weights of GPT2 medium 
    - Prompt type is Alpaca format
    - Used Ollama's llama3 to evaluate the model generated responses


# Steps to Clone and use the repo
- Clone the repo
- Make virtual environment `pyhton3 -m venv myenv`
- Activate the virtual environment `source myenv/bin/activate`
- Now feel free to run and test any-noteboook
