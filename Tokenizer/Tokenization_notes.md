# Tokenization

- Mini LM V2 produces completley different embedding because of a small error
- https://colab.research.google.com/drive/1Gjtm4aXuWFXEXrn65sHpRL7IF-TmFBTZ
- Available tokenizer https://tiktokenizer.vercel.app/?model=gpt-4o
- We observe that simple sentences in urdu like "tum kya kr rhe ho" in urdu script escpicaly has a lot of tokens.
- Higher the number of tokens for a sentence, more capacity is required. If size of tokens inccrease the attention head, it affects the output
- If "hello how are you" takes 7 tokens, and its equivalent in urdu takes 15 tokens, then the performance on urdu will be bad
- Regex patterns are used "in gpt tokenier" to split compnentents fo text whose tokens we donot want to be merged at all. For instance we'd never want world123 maybe to become a single token or like "d1" from world123 to b a single token. Thats why we split the text using regex patterns before tokenization
- GPT-2 an GPT-4 merges and vocab (trained sets) are avaialable in tiktoken library. However, tiktoken doesnt provide code to train tokenizer on a new data.
- Sentence piece however, provides code to train tokenizer on the text too
- Vocabulary size is a hyperparameter. We need to make sure it isnt too small and isnt too large.
- If its too large then probably the tokens generated for a text would be too much which would decrease model performance.
- If its too small then that would mean too many charcaters are squished in a token hence understanding context would be harder
- If for instance (not recommend at all") no of vocab is so small that maybe two words are squished in a single token. Then that is really bad.
- .DefaultCellStyle is a single token in gpt4o cl100k base tokenzier. So when gpt4o is asked how many "l"s are tehre in this word, the model doesnt prodouce the rigt output at once. This shows it is a pretty long token
- One good thing about llama tokenzier over gpt is, it adds a space in teh beginngin so that "world" and " world" are tretaed as the same thing.
- One of the reasons models are not good for non english languages is bacuase of inefficent tokenization of that language
- adding vocab to a tokenzier means that the model would require some modifications. The no. rows of the embedding layers is euqal to the tokenzier size. SUbsequently the forward layers head is also based on the embedding layer. So chnaging the vocab size on a pretrained model means adding row to the embedding layer.
- One technique is gist token paper (referef by karpathy) is to freeze all the weught and just train these new embeddings.

## TO do

- make a gpt-based tokenizer (tiktoken, Andrej karpathay)
- make a llama based tokenizer (sentnece pience library)
- o200k base works good on urdu
- llama doesnt work that good
- cl110k doesnt work that good
- 
- tiktoken cant be trained, study tiktoken and surgery it to use it for using it on your own trained merge and vocab. Merge and vocab can be trained using minbpe (karpathay)
- Make notebook for training, loading, using and saving the model
- SInce urdu caharcters and all characters that we might encounter (emojis, arabic, etc) are not in the 256 single bytes set, make a list of characters and premerge them before training
- Hack around tiktoken
- The native code for regenx tokenizer needs to be optimized (parallel threads for all chunks)
- sentencepiece implementation
- rule based merges, morpholigical tokenizer (or combination), spacy, moses explore

### Can we retrain or add words to a pretrained tokenizer vocabulary?

- Wordpiece based tokenizers (like hugging face ones) allow adding words. BUT you'll have to add whole words. You cannot add subwords.

Tokenziation training is taking too much time.

1. Get stats and 2. merge functions need to be parallelized
2. Either we write the trainer in c++/c
3. Either we write it in pycuda/numba
4. Either we write these two funcs in cuda/c++ and call them in python
5.
