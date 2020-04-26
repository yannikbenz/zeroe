import csv
import json
import logging
import os
import random
import string

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import load_model
from tqdm import tqdm


def should_skip_seq(seq):
    """

    :param seq:
    :return:
    """
    if len(seq) > MAX_DICT_WORD_LEN:
        return True
    if len(seq) < MIN_DICT_WORD_LEN:
        return True
    return False


def load_clean_dictionaries():
    """
    is loading the combilex data into two dictionaries
        word2phone and phone2word

    :return: g2p_dict, p2g_dict
    """
    grapheme_dict = {}
    phonetic_dict = {}

    with open(COMBILEX_PATH, encoding='utf-8') as combilex_file:
        for line in combilex_file:
            # Skip commented lines
            if line[0:3] == ';;;':
                continue

            word, phone = line.strip().split('\t')

            if not should_skip_seq(word):
                if word not in grapheme_dict:
                    grapheme_dict[word] = []
                grapheme_dict[word].append(phone)

            if not should_skip_seq(phone):
                if phone not in phonetic_dict:
                    phonetic_dict[phone] = []
                phonetic_dict[phone].append(word)

    return grapheme_dict, phonetic_dict


def char_and_phone_list():
    allowed_symbols = [".", "-", "'"]
    uppercase_letters = list(string.ascii_uppercase)
    lowercase_letters = list(string.ascii_lowercase)
    char_list = [START_SYM, END_SYM] + uppercase_letters + lowercase_letters
    #
    phone_list = [START_SYM, END_SYM]
    with open(COMBILEX_PATH) as file:
        for line in file:
            grapheme, phoneme = line.split('\t')
            for gp, ph in zip(grapheme, phoneme):
                if gp not in char_list:
                    char_list.append(gp.strip())
                if ph not in phone_list:
                    phone_list.append(ph.strip())
    return [''] + char_list, [''] + phone_list


def id_mappings_from_list(str_list):
    str_to_id = {s: i for i, s in enumerate(str_list)}
    id_to_str = {i: s for i, s in enumerate(str_list)}
    return str_to_id, id_to_str


def g2p_dataset_to_1_hot_tensors():
    char_seqs = []
    phone_seqs = []

    for word, pronuns in word2phonetic_dict.items():
        word_matrix = np.zeros((G2P_MAX_CHAR_SEQ_LEN, CHAR_TOKEN_COUNT))
        for t, char in enumerate(word):
            word_matrix[t, :] = char_to_1_hot(char)
        for pronun in pronuns:
            pronun_matrix = np.zeros((G2P_MAX_PHONE_SEQ_LEN, PHONE_TOKEN_COUNT))
            phones = [START_SYM] + list(pronun) + [END_SYM]
            for t, phone in enumerate(phones):
                pronun_matrix[t, :] = phone_to_1_hot(phone)

            char_seqs.append(word_matrix)
            phone_seqs.append(pronun_matrix)

    return np.array(char_seqs), np.array(phone_seqs)


def p2g_dataset_to_1_hot_tensors():
    char_seqs = []
    phone_seqs = []

    for pronoun, words in phonetic2word_dict.items():
        phone_matrix = np.zeros((P2G_MAX_PHONE_SEQ_LEN, PHONE_TOKEN_COUNT))
        for t, phone in enumerate(pronoun):
            phone_matrix[t, :] = phone_to_1_hot(phone)
        for word in words:
            word_matrix = np.zeros((P2G_MAX_CHAR_SEQ_LEN, CHAR_TOKEN_COUNT))
            word = [START_SYM] + list(word) + [END_SYM]
            for t, char in enumerate(word):
                word_matrix[t, :] = char_to_1_hot(char)

            char_seqs.append(word_matrix)
            phone_seqs.append(phone_matrix)

    return np.array(char_seqs), np.array(phone_seqs)


def char_to_1_hot(char):
    char_id = char_to_id[char]
    hot_vec = np.zeros((CHAR_TOKEN_COUNT))
    hot_vec[char_id] = 1.
    return hot_vec


def phone_to_1_hot(phone):
    phone_id = phone_to_id[phone]
    hot_vec = np.zeros((PHONE_TOKEN_COUNT))
    hot_vec[phone_id] = 1.
    return hot_vec


def baseline_model(encoder_input_token_count, decoder_input_token_count, hidden_nodes=256):
    # Shared Components - Encoder
    char_inputs = Input(shape=(None, encoder_input_token_count))
    encoder = LSTM(hidden_nodes, return_state=True)

    # Shared Components - Decoder
    phone_inputs = Input(shape=(None, decoder_input_token_count))
    decoder = LSTM(hidden_nodes, return_sequences=True, return_state=True)
    decoder_dense = Dense(decoder_input_token_count, activation='softmax')

    # Training Model
    _, state_h, state_c = encoder(char_inputs)  # notice encoder outputs are ignored
    encoder_states = [state_h, state_c]
    decoder_outputs, _, _ = decoder(phone_inputs, initial_state=encoder_states)
    phone_prediction = decoder_dense(decoder_outputs)

    training_model = Model([char_inputs, phone_inputs], phone_prediction)

    # Testing Model - Encoder
    testing_encoder_model = Model(char_inputs, encoder_states)

    # Testing Model - Decoder
    decoder_state_input_h = Input(shape=(hidden_nodes,))
    decoder_state_input_c = Input(shape=(hidden_nodes,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder(phone_inputs, initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    phone_prediction = decoder_dense(decoder_outputs)

    testing_decoder_model = Model([phone_inputs] + decoder_state_inputs, [phone_prediction] + decoder_states)

    return training_model, testing_encoder_model, testing_decoder_model


def train(model, weights_path, encoder_input, decoder_input, decoder_output):
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', patience=3)

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit([encoder_input, decoder_input], decoder_output,
              batch_size=256,
              epochs=200,
              validation_split=0.2,  # Keras will automatically create a validation set for us
              callbacks=[checkpointer, stopper])


def eval(model, weights_path, encoder_input, decoder_input, decoder_output):
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.eval([encoder_input, decoder_input], decoder_output)


def setup_gpu_share_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def g2p_predict(word, encoder, decoder):
    word_matrix = encode_word(word)
    state_vectors = encoder.predict(word_matrix)

    prev_phone = np.zeros((1, 1, PHONE_TOKEN_COUNT))
    prev_phone[0, 0, phone_to_id[START_SYM]] = 1.

    end_found = False
    pronunciation = ''
    while not end_found:
        decoder_output, h, c = decoder.predict([prev_phone] + state_vectors)

        # Predict the phoneme with the highest probability
        predicted_phone_idx = np.argmax(decoder_output[0, -1, :])
        predicted_phone = id_to_phone[predicted_phone_idx]

        pronunciation += predicted_phone

        if predicted_phone == END_SYM or len(pronunciation.split()) > G2P_MAX_PHONE_SEQ_LEN:
            end_found = True

        # Setup inputs for next time step
        prev_phone = np.zeros((1, 1, PHONE_TOKEN_COUNT))
        prev_phone[0, 0, predicted_phone_idx] = 1.
        state_vectors = [h, c]

    return pronunciation.strip()


def p2g_predict(phonetic, encoder, decoder):
    phonetic_matrix = encode_phone(phonetic)
    state_vectors = encoder.predict(phonetic_matrix)

    prev_char = np.zeros((1, 1, CHAR_TOKEN_COUNT))
    prev_char[0, 0, char_to_id[START_SYM]] = 1.

    end_found = False
    word = ''
    while not end_found:
        decoder_output, h, c = decoder.predict([prev_char] + state_vectors)

        # Predict the phoneme with the highest probability
        predicted_char_idx = np.argmax(decoder_output[0, -1, :])
        predicted_char = id_to_char[predicted_char_idx]

        word += predicted_char

        if predicted_char == END_SYM or len(word) > G2P_MAX_PHONE_SEQ_LEN:
            end_found = True

        # Setup inputs for next time step
        prev_char = np.zeros((1, 1, CHAR_TOKEN_COUNT))
        prev_char[0, 0, predicted_char_idx] = 1.
        state_vectors = [h, c]

    return word.strip()


# Helper method for converting vector representations back into words
def one_hot_matrix_to_word(char_seq):
    word = ''
    for char_vec in char_seq[0]:
        if np.count_nonzero(char_vec) == 0:
            break
        hot_bit_idx = np.argmax(char_vec)
        char = id_to_char[hot_bit_idx]
        word += char
    return word


def one_hot_matrix_to_phone(phone_seq):
    phone = ''
    for phone_vec in phone_seq[0]:
        if np.count_nonzero(phone_vec) == 0:
            break
        hot_bit_idx = np.argmax(phone_vec)
        char = id_to_phone[hot_bit_idx]
        phone += char
    return phone


# Some words have multiple correct pronunciations
# If a prediction matches any correct pronunciation, consider it correct.
def g2p_is_correct(word, test_pronunciation):
    correct_pronuns = word2phonetic_dict[word]
    if test_pronunciation in correct_pronuns:
        return True
    return False


def p2g_is_correct(phone, test_word):
    correct_words = phonetic2word_dict[phone]
    if test_word in correct_words:
        return True
    return False


def encode_phone(phonetic: str):
    """

    :param phonetic:
    :return:
    """
    phone_seq = []
    phone_matrix = np.zeros((P2G_MAX_PHONE_SEQ_LEN, PHONE_TOKEN_COUNT))
    for t, phone in enumerate(phonetic):
        if t >= P2G_MAX_PHONE_SEQ_LEN:
            break
        phone_matrix[t, :] = phone_to_1_hot(phone)
    phone_seq.append(phone_matrix)
    return np.array(phone_seq)


def encode_word(word: str):
    """

    :param word:
    :param text:
    :return:
    """
    char_seq = []
    word_matrix = np.zeros((G2P_MAX_CHAR_SEQ_LEN, CHAR_TOKEN_COUNT))
    for t, char in enumerate(word):
        if t >= G2P_MAX_CHAR_SEQ_LEN:
            break
        word_matrix[t, :] = char_to_1_hot(char)
    char_seq.append(word_matrix)
    return np.array(char_seq)


def perturb_word(word: str, skip_stopwords=True, skip_punctuation=True):
    """

    :param skip_stopwords:
    :param word:
    :return: pertubed word,  None if no perturbation is available
    """
    if skip_stopwords and word.lower() in stopwords.words('english') \
            or skip_punctuation and not word.lower().isalnum():
        return None
    try:
        phoneme = g2p_predict(word, g2p_testing_encoder_model, g2p_testing_decoder_model)
        grapheme = p2g_predict(phoneme, p2g_testing_encoder_model, p2g_testing_decoder_model)

        original_word = word_sim_tokenizer.texts_to_sequences([word])
        perturbed_word = word_sim_tokenizer.texts_to_sequences([grapheme])

        original_word = pad_sequences(original_word, maxlen=20)
        perturbed_word = pad_sequences(perturbed_word, maxlen=20)

        similarity = word_sim_model.predict([original_word, perturbed_word]).argmax()

        if word != grapheme and similarity < 2:
            return grapheme
        return None
    except KeyError:
        return None


def perturb_words(word_list: list, perturb_dict):
    """

    :return:
    """
    csv_dict_cache_path = os.path.join(SCRIPT_DIR, 'cache.csv')
    print("Read word dict from file cache.")
    with open(csv_dict_cache_path, mode='w+') as csv_file:
        reader = csv.reader(csv_file)
        cached_dict = {rows[0]: rows[1] for rows in reader}

    word_dict = {**cached_dict, **perturb_dict}
    words = set(word_list)
    i = 0
    for word in tqdm(words, desc="Perturb words"):
        i += 1
        if i % 1000 == 0:
            cache_dict(csv_dict_cache_path, word_dict)
        if word not in word_dict:
            word_dict[word] = perturb_word(word)

    # Write to file cache
    cache_dict(csv_dict_cache_path, word_dict)
    return word_dict


def cache_dict(csv_dict_cache_path, word_dict):
    """
    writes a dict line by line to a csv

    :param csv_dict_cache_path:
    :param word_dict:
    :return:
    """
    print("Write word dict to file cache.")
    with open(csv_dict_cache_path, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(word_dict.items())


def perturb_series(series: pd.Series, perturb_dict, perturbation_level):
    """

    :param series:
    :return:
    """
    def text_to_perturbed_text(text, perturb_dict, perturbation_level):
        words = word_tokenize(text)
        word_indexes = list(range(0, len(words)))
        perturbed_words = 0
        perturb_target = len(words) * perturbation_level

        while perturbed_words < perturb_target:
            if len(word_indexes) < 1:
                break
            index = np.random.choice(word_indexes)
            word_indexes.remove(index)
            word = words[index]
            perturbation = perturb_dict.get(word)
            words[index] = word if perturbation is None else perturbation
            perturbed_words += 1 if perturbation != word else 0
        return TreebankWordDetokenizer().detokenize(tokens=words)

    sentences = series.drop_duplicates().dropna().tolist()
    words = []
    for sentence in tqdm(sentences, desc="Load words of each sentence into dict"):
        for word in word_tokenize(sentence):
            if word not in words:
                words.append(word)

    word2perturbed_dict = perturb_words(words, perturb_dict)
    return series.apply(text_to_perturbed_text, perturb_dict=word2perturbed_dict, perturbation_level=perturbation_level), word2perturbed_dict


# MAIN

# init logger
log = logging.getLogger()
log.info("Starting phonetic pertubator.")

# Setup GPU Share Option to avoid allocating all memory at all
setup_gpu_share_config()

# Download nltk stopwords
nltk.download('stopwords')
nltk.download('punkt')

# If training is enabled for the different models
TRAIN_G2P = False
TRAIN_P2G = False
PRINT_INFO = False

# Setting a limit now simplifies training our model later
MAX_DICT_WORD_LEN = 20
MIN_DICT_WORD_LEN = 2

# training/dictionary data
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
COMBILEX_PATH = os.path.join(SCRIPT_DIR, '../../data/g2pp2g/combilex_en_ph.tsv')
log.info("Vocab file is located @", COMBILEX_PATH)

# out start and end token
# each prediction will start with a \t and end with a \n
START_SYM = '\t'
END_SYM = '\n'

# load two dictionaries
#   1. g2p: translates graphemes to its respective phonemes
#   2. p2g: translates its phonemes to its respective graphemes
log.info("Load grapheme2phoneme & phoneme2grapheme dictionaries.")
word2phonetic_dict, phonetic2word_dict = load_clean_dictionaries()
# the total count of exampels for each respective task
g2p_example_count = np.sum([len(prons) for _, prons in word2phonetic_dict.items()])
p2g_example_count = np.sum([len(word) for _, word in phonetic2word_dict.items()])

# so we can set the log level
if PRINT_INFO:
    log.info("[G2P]", "\n[G2P] ".join(
        [k + ' --> ' + word2phonetic_dict[k][0] for k in random.sample(list(word2phonetic_dict.keys()), 5)]))
    log.info("[P2G]", "\n[P2G] ".join(
        [k + ' --> ' + phonetic2word_dict[k][0] for k in random.sample(list(phonetic2word_dict.keys()), 5)]))

# Load word and phonetic symbols
char_list, phone_list = char_and_phone_list()

# Create ID mappings
phone_to_id, id_to_phone = id_mappings_from_list(phone_list)
char_to_id, id_to_char = id_mappings_from_list(char_list)

if PRINT_INFO:
    # Example:
    print('Char to id mapping: \n', char_to_id)
    print('Phon to id mapping: \n', phone_to_id)

CHAR_TOKEN_COUNT = len(char_to_id) + 1
PHONE_TOKEN_COUNT = len(phone_to_id) + 1

if PRINT_INFO:
    print("Total count of char tokens", CHAR_TOKEN_COUNT)
    print("Total count of phone tokens", PHONE_TOKEN_COUNT)

    # Example:
    print('Char "A" is represented by:\n', char_to_1_hot('A'), '\n-----')
    print('Phone "F" is represented by:\n', phone_to_1_hot('F'))

# + 2 to account for the start & end tokens we need to add
G2P_MAX_CHAR_SEQ_LEN = max([len(word) for word, _ in word2phonetic_dict.items()])
G2P_MAX_PHONE_SEQ_LEN = max([max([len(pron) for pron in pronuns]) for _, pronuns in word2phonetic_dict.items()]) + 2
P2G_MAX_PHONE_SEQ_LEN = max([len(phone) for phone, _ in phonetic2word_dict.items()])
P2G_MAX_CHAR_SEQ_LEN = max([max([len(word) for word in words]) for _, words in phonetic2word_dict.items()]) + 2

# === BASELINE MODEL ===
G2P_BASELINE_MODEL_WEIGHTS = os.path.join(SCRIPT_DIR,
                                          '../../models', 'g2p', 'baseline_model_weights.hdf5')
P2G_BASELINE_MODEL_WEIGHTS = os.path.join(SCRIPT_DIR,
                                          '../../models', 'p2g', 'baseline_model_weights.hdf5')
log.info("Build G2P model")
g2p_training_model, g2p_testing_encoder_model, g2p_testing_decoder_model = baseline_model(CHAR_TOKEN_COUNT,
                                                                                          PHONE_TOKEN_COUNT)
log.info("Build P2G model")
p2g_training_model, p2g_testing_encoder_model, p2g_testing_decoder_model = baseline_model(PHONE_TOKEN_COUNT,
                                                                                          CHAR_TOKEN_COUNT)

# define train steps if necessary
TEST_SIZE = 0.2
if TRAIN_G2P:
    log.info("Train G2P model")

    g2p_char_seq_matrix, g2p_phone_seq_matrix = g2p_dataset_to_1_hot_tensors()

    if PRINT_INFO:
        print('G2P Word Matrix Shape: ', g2p_char_seq_matrix.shape)
        print('G2P Pronunciation Matrix Shape: ', g2p_phone_seq_matrix.shape)

    g2p_phone_seq_matrix_decoder_output = np.pad(g2p_phone_seq_matrix, ((0, 0), (0, 1), (0, 0)), mode='constant')[:, 1:,
                                          :]
    (g2p_char_input_train, g2p_char_input_test,
     g2p_phone_input_train, g2p_phone_input_test,
     g2p_phone_output_train, g2p_phone_output_test) = train_test_split(
        g2p_char_seq_matrix, g2p_phone_seq_matrix, g2p_phone_seq_matrix_decoder_output,
        test_size=TEST_SIZE, random_state=42)
    train(g2p_training_model, G2P_BASELINE_MODEL_WEIGHTS, g2p_char_input_train, g2p_phone_input_train,
          g2p_phone_output_train)
    G2P_TEST_EXAMPLE_COUNT = g2p_char_input_test.shape[0]

if TRAIN_P2G:
    log.info("Train P2G model")

    p2g_char_seq_matrix, p2g_phone_seq_matrix = p2g_dataset_to_1_hot_tensors()

    if PRINT_INFO:
        print('P2G Word Matrix Shape: ', p2g_char_seq_matrix.shape)
        print('P2G Pronunciation Matrix Shape: ', p2g_phone_seq_matrix.shape)

    p2g_char_seq_matrix_decoder_output = np.pad(p2g_char_seq_matrix, ((0, 0), (0, 1), (0, 0)), mode='constant')[:, 1:,
                                         :]
    (p2g_phone_input_train, p2g_phone_input_test,
     p2g_char_input_train, p2g_char_input_test,
     p2g_char_output_train, p2g_char_output_test) = train_test_split(
        p2g_phone_seq_matrix, p2g_char_seq_matrix, p2g_char_seq_matrix_decoder_output,
        test_size=TEST_SIZE, random_state=42)
    train(p2g_training_model, P2G_BASELINE_MODEL_WEIGHTS, p2g_phone_input_train, p2g_char_input_train,
          p2g_char_output_train)
    P2G_TEST_EXAMPLE_COUNT = p2g_phone_input_test.shape[0]

log.info("Load saved weights from", G2P_BASELINE_MODEL_WEIGHTS)
g2p_training_model.load_weights(G2P_BASELINE_MODEL_WEIGHTS)
log.info("Load saved weights from", P2G_BASELINE_MODEL_WEIGHTS)
p2g_training_model.load_weights(P2G_BASELINE_MODEL_WEIGHTS)

#
log.info("Load WordSim model")
WORD_SIM_MODEL_WEIGHTS = os.path.join(SCRIPT_DIR, '../../models', "word_sim", "model.h5")
WORD_SIM_TOKENIZER = os.path.join(SCRIPT_DIR, '../../models', "word_sim", "tokenizer.json")

word_sim_model = load_model(WORD_SIM_MODEL_WEIGHTS)
with open(WORD_SIM_TOKENIZER, 'r') as json_file:
    tokenizer_json = json.load(json_file)
word_sim_tokenizer = tokenizer_from_json(tokenizer_json)
# END
log.info("Finished initialization.")
