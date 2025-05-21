from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, BertTokenizer
from collections import OrderedDict
import numpy as np
import torch

class Evaluator:
    def __init__(self, metrics=None):
        if not metrics:
            metrics = ["rouge", "sacre_bleu", "bertscore", "factkb"]
        self.metrics = metrics
    
    def evaluate(self, predictions, references, documents, metrics=["rouge", "bertscore", "factkb"]):
        result_dict = OrderedDict()
        if "rouge" in metrics:
            rouge_dict = self.calculate_rouge(predictions, references)
            for k, v in rouge_dict.items():
                result_dict[k] = v
        if "sacre_bleu" in metrics:
            sacre_bleu_dict = self.calculate_sacrebleu(predictions, references)
            for k, v in sacre_bleu_dict.items():
                result_dict[k] = v
        if "bertscore" in metrics:
            bertscore_dict = self.calculate_bertscore(predictions, references)
            for k, v in bertscore_dict.items():
                result_dict[k] = v
        if "factkb" in metrics:
            result_dict["factkb"] = self.calculate_factkb(predictions, documents)
            
        for k, v in result_dict.items():
            print(f"{k} -> {v*100:.2f}")
        return result_dict

    def calculate_rouge(self, predictions, references):
        from torchmetrics.functional.text.rouge import rouge_score
        rouge_dict = rouge_score(preds=predictions, target=references)
        return {k: v.item() for k, v in rouge_dict.items()}

    def calculate_sacrebleu(self, predictions, references):
        from torchmetrics.functional.text import sacre_bleu_score
        score = sacre_bleu_score(preds=predictions, target=[[i] for i in references])
        return {"sacre_bleu": score.item()}

    def calculate_bertscore(self, predictions, references):
        import evaluate
        bertscore = evaluate.load("bertscore")
        bertscore_dict = bertscore.compute(predictions=predictions, references=references, model_type="roberta-large-mnli")
        res = {"bertscore_precision": np.mean(bertscore_dict["precision"]), "bertscore_recall": np.mean(bertscore_dict["recall"]), "bertscore_f1": np.mean(bertscore_dict["f1"])}
        return {k: v.item() for k, v in res.items()}

    def calculate_factkb(self, predictions, documents, device):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            "bunsenfeng/FactKB",
             torch_dtype=torch.float16,
             device_map="auto",
            )
        res = []
        device = next(iter(model.parameters())) .device.type
        for i in range(len(predictions)):
            input_pretokenized = f"{predictions[i]} {tokenizer.sep_token} {documents[i]}"
            tokenized_input = tokenizer(input_pretokenized, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                output = model(input_ids=tokenized_input.input_ids.to(device))
            logits = torch.softmax(output.logits, dim=1)  # (bz, 2)
            res.append(logits.squeeze()[-1].item())
        return np.mean(res)    