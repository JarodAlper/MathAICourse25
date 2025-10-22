from transformers import GPT2Tokenizer

# Load the standard GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize the equation

test = "cohomology"
print(f"Tokens for {test}:", tokenizer.tokenize(test))
print("Token IDs:", tokenizer.encode(test))

