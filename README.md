<h1>GPT-Tokenizer</h1>
<p>This repository contains a minimal, clean implementation of the Byte Pair Encoding (BPE) and SentencePiece algorithm, which are commonly used for tokenization in large language models (LLMs). The BPE algorithm operates at the byte level, processing UTF-8 encoded strings. This algorithm was popularized for LLMs by the <a href="https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">GPT-2</a> paper.
SentencePiece operates at the subword level.It is particularly useful for handling out-of-vocabulary words and rare words in a robust manner. Today, all modern LLMs (e.g. GPT, Llama, Mistral) use these algorithm to train their tokenizers.</p>

<p>There are two Tokenizers in this repository, both of which can perform the 3 primary functions of a Tokenizer: 1) train the tokenizer vocabulary and merges on a given text, 2) encode from text to tokens, 3) decode from tokens to text. The files of the repo are as follows:</p>

<p>1.<a href="https://github.com/tanishkaa19/GPT-Tokenizer/blob/main/minbpe/base.py">minbpe/base.py</a>: Implements the Tokenizer class, which is the base class. It contains the train, encode, and decode stubs, save/load functionality", and there are also a few common utility functions. This class is not meant to be used directly, but rather to be inherited from.</p>
<p>2.<a href="https://github.com/tanishkaa19/GPT-Tokenizer/blob/main/minbpe/basic.py">minbpe/basic.py</a>: Implements the BasicTokenizer, the simplest implementation of the BPE algorithm that runs directly on text.</p>
<p>3.<a href="https://github.com/tanishkaa19/GPT-Tokenizer/blob/main/minbpe/my_regex.py">minbpe/my_regex.py</a>: Implements the RegexTokenizer that further splits the input text by a regex pattern, which is a preprocessing stage that splits up the input text by categories (think: letters, numbers, punctuation) before tokenization. This ensures that no merges will happen across category boundaries. This was introduced in the GPT-2 paper and continues to be in use as of GPT-This class also handles special tokens, if any.</p>
<p>4.<a href="https://github.com/tanishkaa19/GPT-Tokenizer/blob/main/minbpe/gpt4.py">minbpe/gpt4.py</a>: Implements the GPT4Tokenizer. This class is a light wrapper around the RegexTokenizer (2, above) that exactly reproduces the tokenization of GPT-4 in the tiktoken library. The wrapping handles some details around recovering the exact merges in the tokenizer, and the handling of some unfortunate (and likely historical?) 1-byte token permutations.</p>
<p>5.<a href="https://github.com/tanishkaa19/GPT-Tokenizer/blob/main/SPM/train_spm">SPM/train_spm</a>: Implements the `GPT4Tokenizer` class. This is a custom tokenizer that utilizes the SentencePiece algorithm for tokenization, tailored specifically for the GPT-4 language model. The class is a wrapper around the SentencePiece library, handling the details of training the SentencePiece model, encoding and decoding text, and managing special tokens unique to GPT-4. It also provides functionality to save the vocabulary to a file for future reference.</p>

<p>Finally, the script <a href="https://github.com/tanishkaa19/GPT-Tokenizer/blob/main/minbpe/train.py">train.py</a> trains the two major tokenizers on the input text<a href="https://github.com/tanishkaa19/GPT-Tokenizer/blob/main/taylorswift.txt"> taylorswift.txt</a> (this is the Wikipedia entry for her kek) and saves the vocab to disk for visualization.</p>
<br>
<h4>You can find the YouTube link for a detailed explanation <a href="https://youtu.be/g82MlcU60Iw?si=whNyR1r2u3MoN5L8">here!</a></h4>
<h2>Contributor</h2>
<a href="https://github.com/KanchiSharma13">Kanchi Sharma</a>


