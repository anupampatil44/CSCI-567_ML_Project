import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import AutoModelForCausalLM , AutoTokenizer, Trainer, TrainingArguments,DefaultDataCollator
import sys
from sklearn.model_selection import train_test_split
import os
import torch


print('No. of GPUs:',torch.cuda.device_count())
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["WANDB_PROJECT"]="sarcasm_generation_through_question_answering"
os.environ["WANDB_LOG_MODEL"]="checkpoint"

data = pd.read_csv('clean_combined_sarcasm_dataset.csv')

data['cleaned_generated_question'].fillna('What\'s your take on this?', inplace=True)

data=data.dropna()

model_name = str(sys.argv[1])
batch_size = int(sys.argv[2])


# Define the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name,padding=True,max_length=300)
tokenizer.pad_token = tokenizer.eos_token

# Handle padding if not present in the tokenizer
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Define a function to preprocess the data
def preprocess_function(examples):
    sarcastic_comments = [str(ex) for ex in examples["Sarcastic Comment"].tolist()]
    context = [ex for ex in examples["Context"].tolist()]
    prompt = [ex for ex in examples["cleaned_generated_question"].tolist()]

    inputs=[i+" "+j for i,j in zip(context,prompt)]
    
    # Display a sample of context and sarcastic comments
    print("Sample Context:", context[:10])
    print("Sample Sarcastic Comments:", sarcastic_comments[:10])
    
    # Tokenize inputs and labels
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(sarcastic_comments, max_length=128, truncation=True, padding=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the data
dataset = preprocess_function(data)

# Display a sample of the preprocessed dataset
# print("Sample Preprocessed Dataset:", dataset[:1])

# Define a custom dataset class
class SarcasticCommentDataset(torch.utils.data.Dataset):
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
training_dataset = SarcasticCommentDataset(dataset["input_ids"], dataset["labels"])




# # Create the dataset
# dataset = SarcasticCommentDataset(
#     contexts=data['Context'].tolist(),
#     prompts=data['cleaned_generated_question'].tolist(),
#     targets=data['Sarcastic Comment'].tolist(),
#     tokenizer=tokenizer,
#     max_length=300
# )

epochs=7

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    logging_dir='./logs',
    push_to_hub=True,
    report_to="wandb",
    fp16=True,
)


train_data, eval_data = train_test_split(training_dataset, test_size=0.1)

# Train the model using the Trainer class
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=DefaultDataCollator(),
)

trainer.train()

# Save the trained model
if len(model_name.split("/"))>1:
    save_name=model_name.split("/")[1]
else:
    save_name=model_name.split("/")[0]
torch.save(model.state_dict(), f"QA_{save_name}_epoch_{epochs}.pth")
