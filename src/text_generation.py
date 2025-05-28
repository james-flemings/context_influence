import argparse
import numpy as np
import torch
from tqdm import tqdm 
from datasets import load_dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities import pretokenize, template_empty_input, template_input
from context_influence import context_influence_decoding
from evaluator import Evaluator
import os
from tqdm import tqdm
import pandas as pd 

def main(args):
    model_n = args.model_name.split('/')[-1]
    dataset_n = args.dataset_name.split('/')[-1]
    results_dir = "results_copy"
    os.makedirs(results_dir, exist_ok=True)
    file_name = f'{dataset_n}_{model_n}_{args.lambd}.csv'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                              padding_side="left",
                                              use_fast=True,
                                              token=args.access_token,
                                              )
    if tokenizer.pad_token is None:
        tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id    
    if dataset_n == "cnn_dailymail":
        split = "test[:1000]"
    elif dataset_n == "pubmed_qa":
        split = "train[:1000]"
    else:
        raise Exception(f"{args.dataset_name} not implemented")

    raw_test_set = load_dataset(args.dataset_name, args.subset, split=split)
    test_set = pretokenize(dataset_n,
                           raw_test_set,
                           tokenizer,
                           args.max_input_length)

    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                trust_remote_code=True,
                                                torch_dtype=torch.float16,
                                                token=args.access_token,
                                                device_map="auto",
                                                )
    device = next(iter(model.parameters())).device.type

    responses = []
    references = [] 
    contexts = []
    stop_token_ids = [tokenizer.eos_token_id,
                      tokenizer.pad_token_id
    ]
    for idx, data in tqdm(enumerate(test_set), total=len(test_set)):
        context_unaware_tokenized_input = tokenizer(template_empty_input(data, dataset_n),
                                                    return_tensors="pt",
                                                    padding=True)
        context_aware_tokenized_input = tokenizer(template_input(data, dataset_n),
                                                  return_tensors="pt",
                                                  padding=True)
        with torch.no_grad():
            output = context_influence_decoding(model,
                                                    tokenizer,
                                                    context_aware_tokenized_input.input_ids.to(device),
                                                    context_unaware_tokenized_input.input_ids.to(device),
                                                    lambd=args.lambd,
                                                    temperature=args.temperature,
                                                    max_length=args.max_new_tokens,
                                                    min_length=args.min_new_tokens,
                                                    stop_token_ids=stop_token_ids,
                                                    device=device,
                                                )
        response = tokenizer.decode(output, skip_special_tokens=True)
        responses.append(response)
        references.append(data['summary'])
        contexts.append(data['context'])
    df = pd.DataFrame({'generations': responses,
                       'references': references})
    evaluator = Evaluator()    
    results_dict = evaluator.evaluate(responses, references, contexts)
    print(results_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help="Dataset to use"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        required=True,
        help="Subset for dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="Model to use"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        required=True,
        help="Temperature parameter for decoding"
    )
    parser.add_argument(
        "--lambd",
        type=float,
        default=1.0,
        required=True,
        help="Influence parameter for decoding"
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=0,
        required=True,
        help="Minimum number of tokens to generate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        required=True,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=2048,
        required=True,
        help="Maximum length of contexts"
    )
    parser.add_argument(
        "--num_contexts",
        type=int,
        default=1000,
        required=True,
        help="Number of contexts to calculate"
    )
    parser.add_argument(
        "--access_token",
        type=str,
        required=True,
        help="Your Access token for certain models"
    ) 
    args = parser.parse_args()
    main(args)