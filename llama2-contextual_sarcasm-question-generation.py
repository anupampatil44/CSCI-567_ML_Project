from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd
from tqdm import tqdm
import sys

def process_chunk(chunk_number):
    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
        batch_size=400,
        tokenizer=tokenizer,
    )

    chunk_size = 10000  # Set an appropriate chunk size based on your system's resources
    df = pd.read_csv("train-sarcasm.csv")

    # Choose the number of rows you want to process in total (e.g., 100,000)
    total_rows = 100000

    # Calculate the start and end index for the current chunk
    start_index = chunk_number * chunk_size
    end_index = min((chunk_number + 1) * chunk_size, total_rows)

    # Check if the start index is beyond the length of the dataframe
    if start_index >= len(df):
        print(f"Chunk {chunk_number} exceeds the number of rows in the dataframe.")
        return

    # Adjust end_index if it overshoots the length of the dataframe
    end_index = min(end_index, len(df))

    df = df.loc[start_index:end_index]

    # Create an empty list to store generated texts
    generated_texts = []

    df['generated_question'] = ''

    # Iterate over the rows in the current chunk
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Get the text from the 'txt' column
        input_text = row['Context']
        sr = row['Sarcastic Comment']

        prompt = f"""<s>[INST] <<SYS>> I will be giving you a context for a statement followed by a sarcastic response.
        As a standup comedian/internet troller who is trying to find the best question to pose over the context and the sarcastic response, what kind of QUESTION should I pose to the audience? Just give the question you generated as your
        response nothing else. Do NOT give any affirmation message, JUST THE GENERATED QUESTION ONLY.<<SYS>>
        {input_text}
        {sr}
        [/INST]
        """
        # Generate text using the model
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=400,
            return_full_text=False,
            min_new_tokens=70,
        )

        # Update the DataFrame with the generated text in each iteration
        df.at[index, 'generated_question'] = sequences[0]['generated_text']

    # Save the results for the current chunk
    df.to_csv(f'sarcasm_updated_annotations_chunk_{chunk_number}.csv', index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <chunk_number>")
        sys.exit(1)

    chunk_number = int(sys.argv[1])
    process_chunk(chunk_number)
