#  Copyright (c) 2020.
#
#  Author: Yannik Benz
#
#  Major parts of this file have been taken from:
#  https://github.com/huggingface/transformers/blob/master/examples/ner/utils_ner.py
#  and were slightly modified to match the respective task
import copy
import json
import os
import logging

from transformers import BertTokenizer, RobertaTokenizer
from transformers.data.processors.utils import InputExample, InputFeatures

logger = logging.getLogger(__name__)


def read_examples_from_file(data_dir, mode, perturber, level):
    """
    TODO: docs

    :param data_dir:
    :param mode:
    :param perturber:
    :param level:
    :return:
    """
    # We want to load the clean examples if either perturber nor level is specified
    # otherwise the perturbed data is loaded for train/dev/test
    if perturber is None or level is None:
        file_path = os.path.join(data_dir, f"{mode}.txt")
    else:
        file_path = os.path.join(data_dir, f"{mode}_{perturber}_{level}.txt")
    logger.info("Read data from file %s", file_path)
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        next(f)  # skip header row
        for line in f:
            splits = line.split(sep="\t")
            label = splits[0]
            sentence1 = splits[1]
            sentence2 = splits[2]
            if label not in ["neutral", "entailment", "contradiction"]:
                continue
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), text_a=sentence1, text_b=sentence2, label=label))
            guid_index += 1
    return examples


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b,
                                       add_special_tokens=True, max_length=max_seq_length)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(attention_mask) == max_seq_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_seq_length
        )
        assert len(token_type_ids) == max_seq_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_seq_length
        )

        label = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("sentences %s" % tokenizer.decode(input_ids))  # skip_special_tokens=True))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label)
        )
    return features


def get_labels(path=None):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        return labels
    else:
        return ["neutral", "entailment", "contradiction"]


if __name__ == '__main__':
    examples = read_examples_from_file("../../../data/datasets/nli", "test", level=None, perturber=None)

    label_list = ["neutral", "entailment", "contradiction"]
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    features_bert = convert_examples_to_features(examples, label_list, max_seq_length=128, tokenizer=tokenizer)
    features_roberta = convert_examples_to_features(examples, label_list, max_seq_length=128, tokenizer=roberta_tokenizer)
    print("Done")
