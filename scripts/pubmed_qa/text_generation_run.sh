for model_name in "facebook/opt-1.3b"
do
    for lambd in 1.0
    do
    CUDA_VISIBLE_DEVICES=6 python src/text_generation.py \
    --dataset_name qiaojin/PubMedQA \
    --subset pqa_labeled \
    --model_name $model_name \
    --temperature 0.8 \
    --lambd $lambd \
    --min_new_tokens 10 \
    --max_new_tokens 50 \
    --max_input_length 2048 \
    --num_contexts 1000 \
    --access_token $HF_ACCESS_TOKEN
    done
done