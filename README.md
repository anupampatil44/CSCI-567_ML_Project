# CSCI-567_ML_Project

Our primary goal is to incorporate commonsense knowledge into the model to create more realistic and contextually appropriate sarcastic outputs. We aim to build a model that produces sarcastic statements given a context in a question-answering format. To achieve this, we utilize a dataset from Reddit as introduced by Khodak et al. (2017), containing 100k sarcastic comments along with their context, to train the AI model. Unlike traditional methods, we generate questions from the context over a pre-trained LLM to train the model. This results in improved performance in capturing sarcasm nuances compared to training with context alone.

This repository contains the following modules(folders) that contains scripts accordingly:
1. Dataset:<br>
   a. Pre-processing modules for the Reddit and IMDB datasets.<br>
   b. Modified dataset containing the questions e generated based on context to get better results. It uses a Llama-2 script to augment the dataset.<br>
3. Sarcasm Detection: We built a separate model for sarcasm detection as a baseline.  
4. Sarcasm Generation:<br>
   a. `Prompt-based-approach.ipynb` - Our prompt-based approach which uses Mistral-7b.<br>
   b. `Sarcasm_GPT2_training_&_inference.ipynb` - Finetuned version of GPT-2 (the model we obtained our best results on). You may follow the notebook for the reproduction of results. <br>
   c. `training-causal_lm_sarcasm_generation.py` - Generalised training script for all our baselines as a Causal-LM task.
   d. `training-seq2seq_sarcasm_generation.py` - Generalised training script for all our baselines as a Seq2Seq task.
6. Results: This contains the inference of the generated comments calculated by our detection model along with their accuracy, F-1 scores, precision, and recall.
7. Generated sarcastic comments: CSV file containing all the generated responses by our models.
