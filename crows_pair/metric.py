import os
import csv
import json
import math
import torch
import argparse
import difflib
import logging
import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from collections import defaultdict
from tqdm import tqdm


def read_data(input_file):
    """
    Load data into pandas DataFrame format.
    """
    
    #df_data = pd.DataFrame(columns=['sent1', 'sent2', 'direction', 'bias_type'])

    with open(input_file) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            direction, gold_bias = '_', '_'
            direction = row['stereo_antistereo']
            bias_type = row['bias_type']

            sent1, sent2 = '', ''
            if direction == 'stereo':
                sent1 = row['sent_more']
                sent2 = row['sent_less']
            else:
                sent1 = row['sent_less']
                sent2 = row['sent_more']

            df_item = {'sent1': sent1,
                       'sent2': sent2,
                       'direction': direction,
                       'bias_type': bias_type}
            rows.append(df_item)
            #df_data = df_data.append(df_item, ignore_index=True)
    
    df_data = pd.DataFrame.from_records(rows)
    return df_data


def get_log_prob_unigram(masked_token_ids, token_ids, mask_idx, lm):
    """
    Given a sequence of token ids, with one masked token, return the log probability of the masked token.
    """
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    # get model hidden states
    output = model(masked_token_ids)
    hidden_states = output[0].squeeze(0)
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)

    # we only need log_prob for the MASK tokens
    assert masked_token_ids[0][mask_idx] == mask_id

    hs = hidden_states[mask_idx]
    target_id = token_ids[0][mask_idx]
    log_probs = log_softmax(hs)[target_id]

    return log_probs


def get_span(seq1, seq2):
    """
    This function extract spans that are shared between two sequences.
    """

    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2


def mask_unigram(data, lm, n=1):
    """
    Score each sentence by masking one word at a time.
    The score for a sentence is the sum of log probability of each word in
    the sentence.
    n = n-gram of token that is masked, if n > 1, we mask tokens with overlapping
    n-grams.
    """
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    log_softmax = lm["log_softmax"]
    mask_token = lm["mask_token"]
    uncased = lm["uncased"]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    sent1, sent2 = data["sent1"], data["sent2"]

    if uncased:
        sent1 = sent1.lower()
        sent2 = sent2.lower()

    # tokenize
    sent1_token_ids = tokenizer.encode(sent1, return_tensors='pt')
    sent2_token_ids = tokenizer.encode(sent2, return_tensors='pt')

    # get spans of non-changing tokens
    template1, template2 = get_span(sent1_token_ids[0], sent2_token_ids[0])

    assert len(template1) == len(template2)

    N = len(template1)  # num. of tokens that can be masked
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    
    sent1_log_probs = 0.
    sent2_log_probs = 0.
    total_masked_tokens = 0

    # skipping CLS and SEP tokens, they'll never be masked
    for i in range(1, N-1):
        sent1_masked_token_ids = sent1_token_ids.clone().detach()
        sent2_masked_token_ids = sent2_token_ids.clone().detach()

        sent1_masked_token_ids[0][template1[i]] = mask_id
        sent2_masked_token_ids[0][template2[i]] = mask_id
        total_masked_tokens += 1

        score1 = get_log_prob_unigram(sent1_masked_token_ids, sent1_token_ids, template1[i], lm)
        score2 = get_log_prob_unigram(sent2_masked_token_ids, sent2_token_ids, template2[i], lm)

        sent1_log_probs += score1.item()
        sent2_log_probs += score2.item()

    score = {}
    # average over iterations
    score["sent1_score"] = sent1_log_probs
    score["sent2_score"] = sent2_log_probs

    return score


def evaluate(model, uncased, tokenizer, output_dir, logger=None):
    """
    Evaluate a masked language model using CrowS-Pairs dataset.
    """

    if not logger:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
    
    logger.info("Crows Pair Evaluation started.")
    
    # load data into panda DataFrame
    input_file = "crows_pair/data/crows_pairs_anonymized.csv"
    df_data = read_data(input_file)

    
        
        
        

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    mask_token = tokenizer.mask_token
    log_softmax = torch.nn.LogSoftmax(dim=0)
    vocab = tokenizer.get_vocab()
    #with open(args.lm_model + ".vocab", "w") as f:
    #    f.write(json.dumps(vocab))

    lm = {"model": model,
          "tokenizer": tokenizer,
          "mask_token": mask_token,
          "log_softmax": log_softmax,
          "uncased": uncased
    }

    # score each sentence. 
    # each row in the dataframe has the sentid and score for pro and anti stereo.
    #df_score = pd.DataFrame(columns=['sent_more', 'sent_less', 
    #                                 'sent_more_score', 'sent_less_score',
     #                                'score', 'stereo_antistereo', 'bias_type'])
    rows = []

    total_stereo, total_antistereo = 0, 0
    stereo_score, antistereo_score = 0, 0

    N = 0
    neutral = 0
    total = len(df_data.index)
    with tqdm(total=total) as pbar:
        for index, data in df_data.iterrows():
            direction = data['direction']
            bias = data['bias_type']
            score = mask_unigram(data, lm)

            for stype in score.keys():
                score[stype] = round(score[stype], 3)

            N += 1
            pair_score = 0
            pbar.update(1)
            if score['sent1_score'] == score['sent2_score']:
                neutral += 1
            else:
                if direction == 'stereo':
                    total_stereo += 1
                    if score['sent1_score'] > score['sent2_score']:
                        stereo_score += 1
                        pair_score = 1
                elif direction == 'antistereo':
                    total_antistereo += 1
                    if score['sent2_score'] > score['sent1_score']:
                        antistereo_score += 1
                        pair_score = 1

            sent_more, sent_less = '', ''
            if direction == 'stereo':
                sent_more = data['sent1']
                sent_less = data['sent2']
                sent_more_score = score['sent1_score']
                sent_less_score = score['sent2_score']
            else:
                sent_more = data['sent2']
                sent_less = data['sent1']
                sent_more_score = score['sent2_score']
                sent_less_score = score['sent1_score']
            rows.append({'sent_more': sent_more,
                                        'sent_less': sent_less,
                                        'sent_more_score': sent_more_score,
                                        'sent_less_score': sent_less_score,
                                        'score': pair_score,
                                        'stereo_antistereo': direction,
                                        'bias_type': bias
                                      })
            #df_score = df_score.append({'sent_more': sent_more,
             #                           'sent_less': sent_less,
              #                          'sent_more_score': sent_more_score,
               #                         'sent_less_score': sent_less_score,
                #                        'score': pair_score,
                 #                       'stereo_antistereo': direction,
                  #                      'bias_type': bias
                   #                   }, ignore_index=True)
    df_score = pd.DataFrame.from_records(rows)
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "crows_result.csv")
    df_score.to_csv(output_file)
    logger.info("Crows Pair Evaluation completed.")
    logger.info(f'Total examples: {str(N)}')
    logger.info(f'Metric score: {str(round((stereo_score + antistereo_score) / N, 2))}')
    logger.info(f'Stereotype score: {str(round(stereo_score  / total_stereo, 2))}')
    if antistereo_score != 0:
        logger.info(f'Anti-stereotype score: {str(round(antistereo_score  / total_antistereo, 2))}')
    logger.info(f"Num. neutral: {str(neutral)} {str(round(neutral / N, 2))}")
    logger.info("")
    
    result = {
    "num_examples": N,
        "metric_score": ((stereo_score + antistereo_score) / N),
        "stereo_score": (stereo_score  / total_stereo),
        "anti_stereo_score": (antistereo_score  / total_antistereo),
        "num_neutral": neutral,
        "percentage_neutral": (neutral / N)
    }
    output_file_score = os.path.join(output_dir, "crows_score.json")
    with open(output_file_score, "w") as f:
        json.dump(result, f)
    
    return result


