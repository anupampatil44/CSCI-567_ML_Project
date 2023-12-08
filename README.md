# CSCI-567_ML_Project

Our primary goal is to incorporate commonsense knowledge into the model to create more realistic and contextually appropriate sarcastic outputs. We aim to build a model that produces sarcastic statements given a context in a question-answering format. To achieve this, we utilize a dataset from Reddit as introduced by Khodak et al. (2017), containing 100k sarcastic comments along with their context, to train the AI model. Unlike traditional methods, we generate questions from the context over a pre-trained LLM to train the model. This results in improved performance in capturing sarcasm nuances compared to training with context alone.

This repository contains the following modules:
1. Dataset:
   a. Pre-processing modules for the Reddit and IMDB datasets.
   b. Modified dataset containing the questions e generated based on context to get better results. Uses a Llama-2 script to augment the dataset.
3. Sarcasm Detection: We built a separate model for sarcasm detection as a baseline.  
4. Sarcasm Generation:
   a. "Prompt-based-approach.ipynb" - Mistral-7b prompt-based version.
   b. "Sarcasm_GPT2_training_&_inference.ipynb" - Finetuned version of GPT-2. Contains the 2 approaches using different hyperparameters and tokenizers(casual and seq2seq).
5. Results: This contains the inference of the generated comments as calculated by our detection model along with their accuracy, F-1 scores, precision, and recall.
6. Generated sarcastic comments: Csv file containing all the generated responses by our models.
