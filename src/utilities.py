from datasets import Dataset
from tqdm import tqdm

def xsum_pretokenize(dataset, tokenizer, max_input_length):
    data = {"context": [], "query": [], "summary": []}
    for i, row in tqdm(enumerate(dataset), desc="truncating documents..."):
        trunc_doc = tokenizer.batch_decode(tokenizer(row['document'], return_tensors="pt", max_length=max_input_length,  truncation=True).input_ids, skip_special_tokens=True)[0]
        data['context'].append(trunc_doc)
        data['summary'].append(row['summary'])
        data["query"].append("Summarize the article in one sentence. Summary:")
    return Dataset.from_dict(data)

def cnn_pretokenize(dataset, tokenizer, max_input_length):
    data = {"context": [], "query": [], "summary": []}
    for i, row in tqdm(enumerate(dataset), desc="truncating documents..."):
        trunc_doc = tokenizer.batch_decode(tokenizer(row['article'], return_tensors="pt", max_length=max_input_length,  truncation=True).input_ids, skip_special_tokens=True)[0]
        data['context'].append(trunc_doc)
        data['summary'].append(row['highlights'])
        data['query'].append("Summary of the above news article:")
    return Dataset.from_dict(data)

def pubmedqa_pretokenize(dataset, tokenizer, max_input_length):
    data = {"context": [], "query": [], "summary": []}
    for i, row in tqdm(enumerate(dataset), desc="truncating documents..."):
        context= ''.join(c for c in row['context']['contexts'])
        trunc_doc = tokenizer.batch_decode(tokenizer(context, return_tensors="pt", max_length=max_input_length, truncation=True).input_ids, skip_special_tokens=True)[0]
        data['context'].append(trunc_doc)
        data['summary'].append(row['long_answer'])
        data['query'].append(f"Question: {row['question']}. Answer:")
    return Dataset.from_dict(data)

def pretokenize(dataset_name, dataset, tokenizer, max_input_length):
    if dataset_name == "xsum":
        return xsum_pretokenize(dataset, tokenizer, max_input_length)
    elif dataset_name == "cnn_dailymail":
        return cnn_pretokenize(dataset, tokenizer, max_input_length)
    elif dataset_name == "pubmed_qa":
        return pubmedqa_pretokenize(dataset, tokenizer, max_input_length)
    return None

def template_input(row, dataset):
    if dataset == "xsum" or dataset == "cnn_dailymail":
        return f"Article: {row['context']}. {row['query']}"
    elif dataset == "PubMedQA":
        return f"Document: {row['context']}. {row['query']}"
    else:
        return ""

def template_empty_input(row, dataset):
    if dataset == "xsum" or dataset == "cnn_dailmail":
        return f"Article: . {row['query']}"
    elif dataset == "pubmed_qa":
        return f"Document: . {row['query']}"
    else:
        return ""