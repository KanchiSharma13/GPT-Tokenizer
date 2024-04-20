import sentencepiece as spm
from my_regex import RegexTokenizer

class GPT4TokenizerSP:
    """Tokenizer using SentencePiece for GPT-4."""

    def __init__(self, model_prefix):
        self.sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        self.special_tokens = {
            '': 100257,
            '': 100258,
            '': 100259,
            '': 100260,
            '': 100276
        }

    def encode(self, text):
        encoded_ids = self.sp.encode_as_ids(text)
        return encoded_ids

    def decode(self, ids):
        text = self.sp.decode_ids(ids)
        return text

    def save_vocab(self, vocab_file):
        vocab = {idx: self.sp.id_to_piece(idx) for idx in range(self.sp.get_piece_size())}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                if idx in self.special_tokens.values():
                    f.write(f"[{token}] {idx}\n")
                else:
                    f.write(f"{token} {idx}\n")

# Train SentencePiece model
training_data = ["taylorswift.txt"]  # List of strings representing your text data
spm.SentencePieceTrainer.train(input=training_data, model_prefix='spm_model', vocab_size=4846)

# Create GPT4TokenizerSP instance
tokenizer = GPT4TokenizerSP("spm_model")

# Example usage
input_text = input("Enter text: ")
encoded = tokenizer.encode(input_text)
print("Tokens: ",encoded)
#decoded = tokenizer.decode(encoded)
#print(decoded)
tokenizer.save_vocab("gpt4_vocab.txt")
