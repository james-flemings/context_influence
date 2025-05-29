import argparse
import numpy as np
import torch
from tqdm import tqdm 
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities import pretokenize, template_empty_input, template_input
from context_influence import context_influence_decoding, partition_n_gram, post_calc_influence
from evaluator import Evaluator
import os
from tqdm import tqdm
import pandas as pd 

np.random.seed(42)

def main(args):
    model_n = args.model_name.split('/')[-1]
    dataset_n = args.dataset_name.split('/')[-1]
    results_dir = "results"
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
        split = f"test[:{args.num_contexts}]"
    elif dataset_n == "pubmed_qa":
        split = f"train[:{args.num_contexts}]"
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

    df = pd.read_csv(os.path.join(results_dir, file_name))
    responses = df['generations']

    references = [] 
    contexts = []
    stop_token_ids = [tokenizer.eos_token_id,
                      tokenizer.pad_token_id
    ]
    token_n_gram_infl = []
    N = 0
    batch_size = 8
    for data, response in tqdm(zip(test_set, responses), total=len(test_set)):
        context_unaware_tokenized_input = tokenizer(template_empty_input(data, dataset_n),
                                                    return_tensors="pt",
                                                    padding=True)
        context_aware_tokenized_input = tokenizer(template_input(data, dataset_n),
                                                  return_tensors="pt",
                                                  padding=True)

        if args.n_gram == None:
            ensemble_context_aware_tokenized_input_ids = None
            batch_size = None
        else:
            ensemble, _ = partition_n_gram(data, tokenizer, dataset_n, args.n_gram)
            ensemble_context_aware_tokenized_input = tokenizer(ensemble, return_tensors="pt", padding=True)
            ensemble_context_aware_tokenized_input_ids = ensemble_context_aware_tokenized_input.input_ids.to(device)
        response_tokenized_input = tokenizer(response, return_tensors="pt")
        with torch.no_grad():
            cur_infl = post_calc_influence(model,
                                    tokenizer,
                                    context_aware_tokenized_input.input_ids.to(device),
                                    context_unaware_tokenized_input.input_ids.to(device),
                                    response_tokenized_input.input_ids[:, 1:].to(device),
                                    args.lambd,
                                    args.temperature,
                                    stop_token_ids,
                                    args.min_new_tokens,
                                    batch_size,
                                    ensemble_context_aware_tokenized_input_ids
                                    )
        token_n_gram_infl.append(cur_infl)
        references.append(data['summary'])
        contexts.append(data['context'])
        N = max(N, len(cur_infl[0]))
        break

    print("Max number of n_grams", N)
    response_n_gram_infl = np.zeros([len(token_n_gram_infl), N])
    response_n_gram_infl[:] = np.nan
    for i, tok_resp in enumerate(token_n_gram_infl):
        sum_ngrams = np.nansum(tok_resp, axis=0)
        response_n_gram_infl[i, :len(sum_ngrams)] = sum_ngrams
    model.cpu()
    del model
    estimator_infl = np.nanmean(response_n_gram_infl, axis=0)
    print(f"N-gram size {args.n_gram}\t Max influence: {np.nanmax(estimator_infl)}")
    with open(f'results/{dataset_n}_{model_n}_{args.n_gram}_gram_infl.npy', 'wb') as f:
        np.save(f, response_n_gram_infl)

    evaluator = Evaluator()    
    results_dict = evaluator.evaluate([responses[0]], references, contexts)
    print(responses[0])
    print(references[0])

    df.to_csv(os.path.join(results_dir, file_name))

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
        help="Number of contexts"
    )
    parser.add_argument(
        "--n_gram",
        type=int,
        default=None,
        #required=True,
        help="n-gram size"
    )
    parser.add_argument(
        "--access_token",
        type=str,
        required=True,
        help="Your Access token for certain models"
    ) 
    args = parser.parse_args()
    main(args)