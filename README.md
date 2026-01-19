# Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models 

This is the official repository for the work "Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models." 

## Environment Setup
We used Python 3.10.12 in our implementation. Run the following lines to set up the evironment: 

```bash
python3.10 -m venv venv
source venv/bin/activate
python3.10 -m pip install -r requirements.txt
``` 

## 📊 Experimental Evaluations

Text generations used in the paper can be found in `results`. To reproduce the main results for `CNN-DM` and `PubMedQA`, run the following:

```bash
bash scripts/cnn_dailymail/main_results_run.sh
bash scripts/pubmed_qa/main_results_run.sh
```

If you want to obtain your own text generations for evaluating context privacy leakage on `CNN-DM` and `PubMedQA`, run the following:

```bash
bash scripts/cnn_dailymail/text_generation_run.sh
bash scripts/pubmed_qa/text_generation_run.sh
```

## Citation

If you found this repository useful, please consider citing our work:
```stex
@inproceedings{flemings2025estimating,
  title={Estimating privacy leakage of augmented contextual knowledge in language models},
  author={Flemings, James and Jiang, Bo and Zhang, Wanrong and Takhirov, Zafar and Annavaram, Murali},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={25092--25108},
  year={2025}
}
```
