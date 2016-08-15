import numpy as np
import tensorflow as tf


def get_one_hot_vector (position, size):
    vector = [0] * size
    vector[position] = 1
    return vector


def format_data(data_list, seq_width, src2id, target2id):

    """
    :param data_list: list with the sentences to be used for training/testing
    :param seq_width: list with the lengths of the sentences (needed for the LSTM cell)
    :param src2id: mapping from source words to ids
    :param trgt2id: mapping from target words to ids
    :return: triple of lists containing the input words, expected output words and sequence lengths
    """

    input_data = np.empty([len(data_list), seq_width], dtype=int)
    labels = np.empty([len(data_list), seq_width], dtype=int)
    seq_length = np.empty([len(data_list)], dtype=int)
    empty_embedding = [0] * len(target2id)
    for count, sent in enumerate(data_list):
        if len(sent) > 50:
            sent = sent[:50]
        # Create a [seq_width, vocab_size]-shaped array, pad it with empty vectors when necessary.
        input_padded = [src2id[word] if word in src2id else src2id["UNK"] for word,_ in sent] \
                        + (seq_width - len(sent)) * [0]
        input_array = np.asarray(input_padded)
        input_data[count] = input_array
        labels_padded = [target2id[word] if word in target2id
                         else target2id["UNK"] for _,word in sent] \
                       + (seq_width-len(sent)) * [0]
        #labels_padded = [get_one_hot_vector(target2id[word], len(target2id)) if word in target2id
        #                 else get_one_hot_vector(target2id["UNK"], len(target2id)) for _,word in sent] \
        #               + (seq_width-len(sent)) * [empty_embedding]
        labels_array = np.asarray(labels_padded)
        labels[count] = labels_array
        seq_length[count] = len(sent)
    return input_data, labels, seq_length