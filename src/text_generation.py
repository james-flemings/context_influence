import argparse
import numpy as np
import torch
from tqdm import tqdm 
from datasets import load_dataset

def main(args):
    pass

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
        "--min_new_tokens",
        type=int,
        default=0,
        required=True,
        help="Minimum number of tokens to generate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=str,
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
        "--access_token",
        type=str,
        required=True,
        help="Your Access token for certain models"
    ) 
    args = parser.parse_args()
    main(args)