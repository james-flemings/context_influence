from datasets import load_dataset
from torchmetrics.functional.text.rouge import rouge_score
import pandas as pd
import os 

lambd = 0.5
dataset_name = "cnn"
model = "opt-1.3b"
file_name = f'{dataset_name}_{model}_{lambd}.csv'
dir_name = 'results'

dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test[:1000]")
df = pd.read_csv(os.path.join(dir_name, file_name))

preds = df['generations']
references = dataset['article']

repeat_threshold = 25
repeat_count = 0

rouge_threshold = 0.5
rouge_count = 0

for i in range(1000):
    pred = preds[i]
    ref = references[i]
    repeat = set(pred.split()) & set(ref.split())
    scores = rouge_score(pred, ref)

    if len(repeat) > repeat_threshold:
        repeat_count += 1
    if scores['rougeL_recall'] > rouge_threshold or scores['rougeL_precision'] > rouge_threshold:
        rouge_count += 1

print("Repeat prompts:", repeat_count)
print("Rouge prompts:", rouge_count)