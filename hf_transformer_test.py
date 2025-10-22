import re
from transformers import pipeline

# Prompt for GPT-5 within cursor: Please write a very simply Python script that uses a pre-trained huggingface transformer to predict the next number in a pre-defined sequence [1, 1, 2, 3, 5, 8, 13, 21].  Keep it simple.


def extract_first_integer(text: str) -> str | None:
    match = re.search(r"\d+", text)
    return match.group(0) if match else None


def main() -> None:
    sequence_prompt = "Predict the next number in the sequence:1, 1, 2, 3, 5, 8, 13, 21"

    generator = pipeline(
        task="text-generation",
        # model="mistralai/Mistral-7B-Instruct-v0.3",
        model="gpt2",
        # tokenizer="gpt2",
        device=-1,  # CPU
    )

    result = generator(
        sequence_prompt,
        max_new_tokens=4,
        do_sample=False,  # greedy for determinism
        eos_token_id=generator.tokenizer.eos_token_id,
    )[0]["generated_text"]

    response = result[len(sequence_prompt) :]
    print("Response:", response)
    next_number = extract_first_integer(response)
    print(next_number if next_number is not None else response.strip())


if __name__ == "__main__":
    main()