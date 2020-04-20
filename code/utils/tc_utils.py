#  Copyright (c) 2020.
#
#  Author: Yannik Benz
#

import os
import logging

import csv

from tqdm import tqdm
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
        reader = csv.reader(f, )
        next(reader)  # skip header row
        for i, row in enumerate(reader):
            labels = [int(x) for x in row[2:]]
            comment_text = row[1]
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), text_a=comment_text, label=labels))
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
    features = []
    for (ex_index, example) in tqdm(enumerate(examples), desc="Examples -> Features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        inputs = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_seq_length)
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

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids),
                                                                                           max_seq_length)
        assert len(attention_mask) == max_seq_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_seq_length
        )
        assert len(token_type_ids) == max_seq_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_seq_length
        )

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("sentence %s" % tokenizer.decode(input_ids))  # skip_special_tokens=True))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("labels: %s" % (','.join([str(x) for x in example.label])))

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                          label=example.label)
        )
    return features


def get_labels(path=None):
    return "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
