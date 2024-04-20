# Import the GPT4Tokenizer class from the provided code
from gpt4 import GPT4Tokenizer

def encode_input(tokenizer, input_text):
    # Tokenize the input text using the GPT-4 tokenizer
    token_ids = tokenizer.encode(input_text)
    return token_ids

def main():
    # Create an instance of the GPT4Tokenizer
    tokenizer = GPT4Tokenizer()

    # Prompt the user to input text
    user_input = input("Enter text to encode: ")

    # Encode the user input
    encoded_text = encode_input(tokenizer, user_input)

    # Print the encoded text
    print("Encoded text:")
    print(encoded_text)

if __name__ == "__main__":
    main()
