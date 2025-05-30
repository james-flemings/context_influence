import torch
import torch.nn.functional as F
from utilities import template_empty_input, template_input

def calc_distributions(model, 
                      tokenizer, 
                      context_aware_input_ids, 
                      context_unaware_input_ids,
                      response_input_ids,
                      lambd,
                      temperature,
                      stop_token_ids,
                      min_length,
                      t,
                      batch_size=None,
                      ensemble_context_aware_input_ids=None,
                      ):
    ensemble_proj_output = None
    priv_context_aware_input_ids = torch.cat([context_aware_input_ids,
                                      response_input_ids[:, :t]],
                                     dim=1)
    pub_logit = model(torch.cat([context_unaware_input_ids,
                                 response_input_ids[:, :t]],
                                dim=1)
                     ).logits.squeeze()[-1, :].type(torch.float64)

    priv_logit = model(priv_context_aware_input_ids).logits[-1, -1, :].type(torch.float64)

    if batch_size != None:
        N = ensemble_context_aware_input_ids.shape[0]
        num_batch = N // batch_size + 1 if N % batch_size != 0 else N // batch_size
        ensemble_priv_context_aware_input_ids = torch.cat([ensemble_context_aware_input_ids,
                                  response_input_ids[:, :t].repeat(N, 1)],
                                 dim=1)
        ensemble_priv_logit = torch.cat([model(ensemble_priv_context_aware_input_ids[i*batch_size:(i+1)*batch_size]).logits[:, -1, :].type(torch.float64)#.cpu()
                 for i in range(0, num_batch)], axis=0)
        ensemble_proj_logit = lambd * ensemble_priv_logit + (1-lambd) * pub_logit.repeat(N, 1)

    proj_logit = lambd * priv_logit + (1-lambd) * pub_logit

    if t < min_length:
        pub_logit[stop_token_ids[0]] = -float("Inf")
        proj_logit[stop_token_ids[0]] = -float("Inf")
        if ensemble_context_aware_input_ids != None:
            ensemble_proj_logit[:, stop_token_ids[0]] = -float("Inf")

    if pub_logit.shape[0] > len(tokenizer):
        pub_logit[len(tokenizer):pub_logit.shape[0]] = -float("Inf")
        proj_logit[len(tokenizer):pub_logit.shape[0]] = -float("Inf")
        if ensemble_context_aware_input_ids != None:
            ensemble_proj_logit[:, len(tokenizer):pub_logit.shape[0]] = -float("Inf")

    pub_output = F.softmax(pub_logit / temperature, dim=-1)
    proj_output = F.softmax(proj_logit / temperature, dim=-1)
    if ensemble_context_aware_input_ids != None:
        ensemble_proj_output = F.softmax(ensemble_proj_logit / temperature, dim=-1)
    return proj_output, pub_output, ensemble_proj_output

def calc_n_gram_influence(p, q, idx):
    return abs(torch.log(p[idx]/q[idx])).cpu().numpy()

def calc_context_influence(output, ensemble_outputs, idx):
    return [calc_n_gram_influence(output, ensemble_outputs[i, :], idx)[0][0]
             for i in range(0, ensemble_outputs.shape[0])]

def partition_n_gram(data, tokenizer, dataset_name, n):
    document_ids = tokenizer(data['context']).input_ids
    length = len(document_ids)
    groups = []
    n_grams = []
    N = length - n + 1
    if N < 0:
        return [template_empty_input(data, dataset_name)], n_grams
    for i in range(N):
        removed_n_gram = document_ids[:i] + document_ids[i+n:]
        n_grams.append(document_ids[i:i+n])
        row = {'context': tokenizer.decode(removed_n_gram, skip_special_tokens=True), 'query': data['query']}
        groups.append(template_input(row, dataset_name))
    return groups, n_grams


def context_influence_decoding(model,
                               tokenizer,
                               context_aware_input_ids,
                               context_unaware_input_ids,
                               lambd,
                               temperature,
                               max_length,
                               min_length,
                               stop_token_ids,
                               device,
                               ):
    response_input_ids = torch.LongTensor([[]]).to(device)
    for t in range(max_length):
        proj_output, _, _ = calc_distributions(model, 
                                               tokenizer, 
                                               context_aware_input_ids,
                                               context_unaware_input_ids,
                                               response_input_ids,
                                               lambd,
                                               temperature,
                                               stop_token_ids,
                                               min_length,
                                               t,
                                              )
        pred_idx = proj_output.multinomial(1).view(1, -1).long().to(device)
        if pred_idx.cpu()[0].item() in stop_token_ids:
            break

        response_input_ids = torch.cat([response_input_ids, pred_idx], dim=1)
        del pred_idx
    return response_input_ids.cpu()[0]

def post_calc_influence(model,
                   tokenizer,
                   context_aware_input_ids,
                   context_unaware_input_ids,
                   response_input_ids,
                   lambd,
                   temperature,
                   stop_token_ids,
                   min_length,
                   batch_size=None,
                   ensemble_context_aware_input_ids=None,
                  ):
    infl_vals = []
    for t in range(response_input_ids.shape[1]):
        proj_output, pub_output, ensemble_proj_output = calc_distributions(model,
                                                                          tokenizer, 
                                                                          context_aware_input_ids,
                                                                          context_unaware_input_ids,
                                                                          response_input_ids,
                                                                          lambd,
                                                                          temperature,
                                                                          stop_token_ids,
                                                                          min_length,
                                                                          t,
                                                                          batch_size,
                                                                          ensemble_context_aware_input_ids)
        ids = torch.nonzero(pub_output)
        if ensemble_context_aware_input_ids == None:
            infl_val = [calc_n_gram_influence(proj_output[ids], pub_output[ids], response_input_ids[:, t])[0][0]]
        else:
            infl_val = calc_context_influence(proj_output[ids], ensemble_proj_output[:, ids].squeeze(-1), response_input_ids[:, t])
        infl_vals.append(infl_val)    
    return infl_vals