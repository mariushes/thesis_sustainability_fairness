''' Main script for loading models and running WEAT tests '''

import os
import sys
import random
import re
import argparse
import logging as log
#log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)  # noqa

from csv import DictWriter
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from seat.sentbias.data import (
    load_json, load_encodings, save_encodings, load_jiant_encodings,
)
import seat.sentbias.weat as weat

import seat.sentbias.encoders.bert as bert

TEST_EXT = '.jsonl'



def test_sort_key(test):
    '''
    Return tuple to be used as a sort key for the specified test name.
   Break test name into pieces consisting of the integers in the name
    and the strings in between them.
    '''
    key = ()
    prev_end = 0
    for match in re.finditer(r'\d+', test):
        key = key + (test[prev_end:match.start()], int(match.group(0)))
        prev_end = match.end()
    key = key + (test[prev_end:],)

    return key





def split_comma_and_check(arg_str, allowed_set, item_type):
    ''' Given a comma-separated string of items,
    split on commas and check if all items are in allowed_set.
    item_type is just for the assert message. '''
    items = arg_str.split(',')
    for item in items:
        if item not in allowed_set:
            raise ValueError("Unknown %s: %s!" % (item_type, item))
    return items


def maybe_make_dir(dirname):
    ''' Maybe make directory '''
    os.makedirs(dirname, exist_ok=True)


def run_seat(model, tokenizer, device, seed, output_dir, data_dir="seat/tests", tests="all_sent", dont_cache_encs=True,ignore_cached_encs=True,n_samples=100000,parametric=False,logger=None):

    exp_dir = os.path.join(output_dir,"exp")
    datalog_file = os.path.join(output_dir,"seat_log.log")
    results_path = os.path.join(output_dir,"seat_results.tsv")
    if not logger:
        log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)
        logger = log.getLogger()
        if log_file:
            logger.addHandler(log.FileHandler(log_file))



    if seed >= 0:
        logger.info('Seeding random number generators with {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
    maybe_make_dir(exp_dir)


    all_tests = sorted(
        [
            entry[:-len(TEST_EXT)]
            for entry in os.listdir(data_dir)
            if not entry.startswith('.') and entry.endswith(TEST_EXT)
        ],
        key=test_sort_key
    )
    logger.debug('Tests found:')
    for test in all_tests:
        logger.debug('\t{}'.format(test))

    if tests == "all_sent":
        tests = "sent-weat3,sent-weat3b,sent-weat4,sent-weat5,sent-weat5b,sent-weat6,sent-weat6b,sent-weat7,sent-weat7b,sent-weat8,sent-weat8b,sent-weat9,sent-weat10,sent-angry_black_woman_stereotype,sent-angry_black_woman_stereotype_b,heilman_double_bind_competent_1,heilman_double_bind_competent_1-,heilman_double_bind_competent_one_sentence,heilman_double_bind_competent_one_word,sent-heilman_double_bind_competent_one_word,heilman_double_bind_likable_1,heilman_double_bind_likable_1-,heilman_double_bind_likable_one_sentence,heilman_double_bind_likable_one_word,sent-heilman_double_bind_likable_one_word"
    if tests == "all":
        tests = None

    tests = split_comma_and_check(tests, all_tests, "test") if tests is not None else all_tests
    logger.info('Tests selected:')
    for test in tests:
        log.info('\t{}'.format(test))

    results = []

    for test in tests:
        logger.info('Running test {}.'.format(test))
        enc_file = os.path.join(exp_dir, f"{test}.h5")
        #logger.info('Running test {} for model {}'.format(test, model_name))
        #enc_file = os.path.join(exp_dir, "%s.%s.h5" % ( "%s;%s" % (model_name, model_options) if model_options else model_name, test))
        if not ignore_cached_encs and os.path.isfile(enc_file):
            log.info("Loading encodings from %s", enc_file)
            encs = load_encodings(enc_file)
            encs_targ1 = encs['targ1']
            encs_targ2 = encs['targ2']
            encs_attr1 = encs['attr1']
            encs_attr2 = encs['attr2']
        else:
            # load the test data
            encs = load_json(os.path.join(data_dir, "%s%s" % (test, TEST_EXT)))

            # load the model and do model-specific encoding procedure
            logger.info('Computing sentence encodings')
            encs_targ1 = bert.encode_transformers(model, tokenizer, device, encs["targ1"]["examples"])
            encs_targ2 = bert.encode_transformers(model, tokenizer, device, encs["targ2"]["examples"])
            encs_attr1 = bert.encode_transformers(model, tokenizer, device, encs["attr1"]["examples"])
            encs_attr2 = bert.encode_transformers(model, tokenizer, device, encs["attr2"]["examples"])

            encs["targ1"]["encs"] = encs_targ1
            encs["targ2"]["encs"] = encs_targ2
            encs["attr1"]["encs"] = encs_attr1
            encs["attr2"]["encs"] = encs_attr2

            logger.info("\tDone!")
            if not dont_cache_encs:
                log.info("Saving encodings to %s", enc_file)
                save_encodings(encs, enc_file)

        enc = [e for e in encs["targ1"]['encs'].values()][0]
        d_rep = enc.size if isinstance(enc, np.ndarray) else len(enc)

        # run the test on the encodings
        logger.info("Running SEAT...")
        logger.info("Representation dimension: {}".format(d_rep))
        esize, pval = weat.run_test(encs, n_samples=n_samples, parametric=parametric)
        results.append(dict(
            test=test,
            p_value=pval,
            effect_size=esize,
            num_targ1=len(encs['targ1']['encs']),
            num_targ2=len(encs['targ2']['encs']),
            num_attr1=len(encs['attr1']['encs']),
            num_attr2=len(encs['attr2']['encs'])))

    for r in results:
        logger.info("\tTest {test}:\tp-val: {p_value:.9f}\tesize: {effect_size:.2f}".format(**r))

    if results_path is not None:
        logger.info('Writing results to {}'.format(results_path))
        with open(results_path, 'w') as f:
            writer = DictWriter(f, fieldnames=results[0].keys(), delimiter='\t')
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    return results
