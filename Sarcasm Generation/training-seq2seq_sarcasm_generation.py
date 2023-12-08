import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DefaultDataCollator,Seq2SeqTrainingArguments, Seq2SeqTrainer,AutoModelForSeq2SeqLM
import sys

# Read the combined CSV file
df = pd.read_csv("combined.csv")

# Extract relevant columns from the DataFrame
Comment = df['Sarcastic Comment']
Context = df['Context']
df = df[['Sarcastic Comment', 'Context']]

# Specify the GPT-2 model variant
model_name = str(sys.argv[1])

# Use AutoTokenizer to automatically load the appropriate tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use AutoModelForCausalLM to automatically load the appropriate language model head
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,is_decoder=True)

# Handle padding if not present in the tokenizer
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Define a function to preprocess the data
def preprocess_function(examples):
    sarcastic_comments = [str(ex) for ex in examples["Sarcastic Comment"].tolist()]
    context = [ex for ex in examples["Context"].tolist()]
    
    # Display a sample of context and sarcastic comments
    print("Sample Context:", context[:10])
    print("Sample Sarcastic Comments:", sarcastic_comments[:10])
    
    # Tokenize inputs and labels
    model_inputs = tokenizer(context, max_length=128, truncation=True, padding=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(sarcastic_comments, max_length=128, truncation=True, padding=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the data
dataset = preprocess_function(df)

# Display a sample of the preprocessed dataset
print("Sample Preprocessed Dataset:", dataset[:1])

# Define a custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        index = int(index)
        input_ids = torch.tensor(self.inputs[index]).squeeze()
        target_ids = torch.tensor(self.targets[index]).squeeze()
        return {"input_ids": input_ids, "labels": target_ids}

# Create an instance of the custom dataset
training_dataset = CustomDataset(dataset["input_ids"], dataset["labels"])

# Display a sample from the training dataset
# print("Sample from Training Dataset:", training_dataset[0])

epochs=7

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=int(sys.argv[2]),
    per_device_eval_batch_size=int(sys.argv[2]),
    num_train_epochs=epochs,
    weight_decay=0.01,
    push_to_hub=True,
    predict_with_generate=True  
)

from sklearn.model_selection import train_test_split
train_data, eval_data = train_test_split(training_dataset, test_size=0.1)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=DefaultDataCollator(),
)

# Fine-tune the model
trainer.train()

# Save the trained model
if len(model_name.split("/"))>1:
    save_name=model_name.split("/")[1]
else:
    save_name=model_name.split("/")[0]
torch.save(model.state_dict(), f"{save_name}_epoch_{epochs}.pth")
