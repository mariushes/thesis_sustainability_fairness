import torch

def hard_debiasing_loss(results_S, results_T):

    
    
    return None

def lmd_loss(results_S, results_T, model_S, device):
    base_model_name = "bert-base-uncased"
    bias_type = "gender"
    pad_id = 0
    if base_model_name == "bert-base-uncased":
        target_ids_list = {"gender": [[2450, 2158],
 [2308, 2273],
 [2611, 2879],
 [2388, 2269],
 [2684, 2365],
 [2564, 3129],
 [12286, 7833],
 [3566, 3611],
 [8959, 18087],
 [3203, 10170],
 [21658, 2909],
 [22566, 3677],
 [2931, 3287],
 [5916, 4470],
 [2905, 2567],
 [2016, 2002]]}
        mask_id = 103
    with torch.no_grad():
        target_ids_tensor = torch.tensor(target_ids_list["gender"]).to(device)

        input_ids = results_S["batch"]["input_ids"]
        loss = 0
        counter = 0
        for i_sent, sent in enumerate(input_ids):
            for i, input_id in enumerate(sent):
                for x, target_pair in enumerate(target_ids_tensor):
                    if input_id in target_pair:
                        log_probs1 = 0
                        log_probs2 = 0
                        sent1 = sent.clone()
                        sent2 = sent.clone()

                        for target in target_pair:
                            if target != input_id:
                                sent2[i] = target
                        for idx, token_id in enumerate(sent1):
                            if token_id in target_pair or token_id == pad_id:
                                continue
                            sent1_c = sent1.clone()
                            sent2_c = sent2.clone()
                            sent1_c[idx] = mask_id
                            sent2_c[idx] = mask_id
                            log_probs1 += get_log_prob_unigram(masked_token_ids=sent1_c, token_ids=sent1, mask_idx=idx, model=model_S, mask_id=mask_id)
                            log_probs2 += get_log_prob_unigram(masked_token_ids=sent2_c, token_ids=sent2, mask_idx=idx, model=model_S, mask_id=mask_id)

                            del sent1_c
                            del sent2_c

                        loss += abs(log_probs1 - log_probs2)
                        counter += 1
                        del sent1
                        del sent2
                        torch.cuda.empty_cache()
    if counter > 0:
        loss = loss / counter
    
    return loss
                    
                    
                        
                
                
            

def get_log_prob_unigram(masked_token_ids, token_ids, mask_idx, model, mask_id):
    # Adapted from https://github.com/nyu-mll/crows-pairs/blob/master/metric.py
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    masked_token_ids = masked_token_ids.reshape([1, masked_token_ids.size(dim=0)])
    token_ids = token_ids.reshape([1, token_ids.size(dim=0)])
    
    log_softmax = torch.nn.LogSoftmax(dim=0)

    # get model hidden states
    output = model(masked_token_ids)
    hidden_states = output[0].squeeze(0)

    # we only need log_prob for the MASK tokens
    assert masked_token_ids[0][mask_idx] == mask_id

    hs = hidden_states[mask_idx]
    target_id = token_ids[0][mask_idx]
    log_probs = log_softmax(hs)[target_id]

    return log_probs
                    
        
    