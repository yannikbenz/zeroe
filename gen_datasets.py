#  Copyright (c) 2020.
#
#  Yannik Benz
#
#  Yannik Benz
import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

from code.attacks import simple_attacks as simple

import numpy as np


def perturb_conllu(input_file_path, attacker, perturbation_level, output_file_path):
    """
    :param input_file_path: path to wiki input file
    :param attacker: attacker to apply
    :param perturbation_level: attack level to apply
    :param output_file_path: file to write the perturbed data to
    :param _nrows: (optional) limit the rows
    """
    global phonetic_cache

    words = set()

    with open(input_file_path, encoding='utf-8', mode='r') as f:
        for line in f:
            if line.startswith('#') or line == '' or line == '\n':
                continue
            else:
                words.add(line.split('\t')[0])

    perturbed_word_dict = dict.fromkeys(words)

    if attacker == 'phonetic':
        phonetic_perturbed_words_dict = g2pp2g.perturb_words(list(words), phonetic_cache)
        perturbed_word_dict = {**perturbed_word_dict, **phonetic_perturbed_words_dict}
    else:
        for word in words:
            if attacker == 'viper':
                perturbed_word = viper_ices.run(word, prob=perturbation_level, top_n=20)
            else:
                perturbed_word = simple.simple_perturb(word, attacker, perturbation_level)
            perturbed_word_dict[word] = perturbed_word

    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        with open(input_file_path, 'r', encoding='utf-8') as in_file:
            outlines = []
            sample_count = 0
            sentence = []
            for line in in_file:
                if line.startswith('#') or line == '':  # bos
                    outlines.append(line)
                elif line == '\n':  # eos
                    sample_count += 1
                    perturbed_words = 0
                    word_indexes = list(range(0, len(sentence)))
                    perturb_target = len(sentence) * perturbation_level
                    while perturbed_words < perturb_target:
                        if len(word_indexes) < 1:
                            break
                        index = np.random.choice(word_indexes)
                        word_indexes.remove(index)
                        word = sentence[index][0]
                        perturbed_word = perturbed_word_dict.get(word)
                        if perturbed_word is None:
                            continue
                        sentence[index][0] = perturbed_word
                        perturbed_words += 1 if perturbed_word != word else 0
                    outlines.extend(f"{word}\t{tag}" for (word, tag) in sentence)
                    outlines.append(line)
                    sentence = []  # clear
                else:
                    sentence.append((line.split('\t')))
        out_file.writelines(outlines)


def perturb_series(series: pd.Series, attacker, perturbation_level=0.2):
    """
    :param series:
    :param attacker:
    :param perturbation_level: 0.2 low, 0.8 high
    :return:
    """
    global phonetic_cache
    if attacker == 'viper':
        return series.progress_apply(viper_ices.run, prob=perturbation_level, top_n=20)
    elif attacker == 'phonetic':
        perturbed, phonetic_cache = g2pp2g.perturb_series(series, phonetic_cache,
                                                          perturbation_level=perturbation_level)
        return perturbed
    else:
        return series.progress_apply(simple.simple_perturb, method=attacker, perturbation_level=perturbation_level)


def load_pd_data(data_dir, data_split, eval_task):
    if eval_task == 'snli':
        return pd.read_csv(os.path.join(data_dir, f'{eval_task}/{data_split}.txt'), sep='\t').dropna(
            subset=['sentence1', 'sentence2'])
    elif eval_task == 'tc':
        return pd.read_csv(os.path.join(data_dir, f"{eval_task}/{data_split}.txt"))


if __name__ == '__main__':
    # init logger
    log = logging.getLogger()

    available_methods = ["full-swap", "inner-swap", "intrude", "disemvowel",
                         "truncate", "keyboard-typo", "natural-typo", "segment",
                         "phonetic", "viper"]
    tasks = ["tc", "snli", "pos", "wiki"]
    levels = ["low", "mid", "high"]
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, choices=tasks, required=True, dest='task')
    parser.add_argument('--method', '-m', nargs='+', required=True,
                        choices=available_methods + ["all"], dest='methods')
    parser.add_argument('--sample', '-s', type=bool, default=False, dest='sample')
    parser.add_argument('--level', '-l', type=str, choices=levels, required=True, dest='level')
    parser.add_argument('--indir', '-i', type=str, dest='indir')
    # parser.add_argument('--outdir', '-o', type=str, dest='outdir', default=SCRIPT_DIR)
    args = parser.parse_args()

    # load CLI arguments
    task = args.task
    methods = args.methods
    sample = args.sample
    level = args.level
    indir = args.indir

    if 'all' in methods:
        methods = available_methods

    if 'phonetic' in methods:
        from code.models import g2pp2g

        g2pp2g.setup_gpu_share_config()

    if 'viper' in methods:
        from code.attacks.visual import viper_ices

    # init phonetic cache
    phonetic_cache = {}

    tqdm.pandas()  # make tqdm available for pandas

    if level == 'low':
        pert_level = 0.2
    elif level == 'mid':
        pert_level = 0.5
    elif level == 'high':
        pert_level = 0.8
    else:
        raise ValueError

    save_path = os.path.join(indir, task)

    for method in methods:
        log.info(f"START {method}")
        for split in ["train", "dev", "test"]:
            log.info(f"Perturb {split} split")
            if method == 'segment' and (task == 'pos' or task == 'wiki'):
                print("Segmentation cannot be applied to POS tagging.")
            elif task == 'wiki' or task == 'pos':
                perturb_conllu(os.path.join(indir, task, f'{split}.txt'), method, pert_level,
                               os.path.join(save_path, f'{split}_{method}_{level}.txt'))
            elif task == 'tc':
                if split == 'dev':
                    continue
                tc_data = load_pd_data(indir, split, task)
                tc_pert_series = tc_data.apply(
                    lambda x: perturb_series(x, method, perturbation_level=pert_level)
                    if x.name == 'comment_text' else x, axis=0)
                tc_pert_series.to_csv(os.path.join(save_path, f"{split}_{method}_{level}.txt"), index=None)
            elif task == 'snli':
                snli_data = load_pd_data(indir, split, task)
                snli_pert_series = snli_data.apply(
                    lambda x: perturb_series(x, method, perturbation_level=pert_level)
                    if x.name in ["sentence1", "sentence2"] else x, axis=0)
                snli_data.to_csv(os.path.join(save_path, f"{split}_{method}_{level}.txt"), index=None, sep='\t')
