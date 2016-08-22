import argparse
import random
import os
import pickle

import tensorflow as tf
from tensorflow.python.ops.nn_ops import *
import numpy as np
#import gensim

import word2vec_optimized_utf8 as w2v
import data_reader
import format_data
import dictionary_builder

if __name__ == "__main__":

    parser = argparse.ArgumentParser(version='1.0',description='Train a neural POS tagger.')
    parser.add_argument('-train', dest='train', required=False, default=True,
                        help='Indicate whether the model shoul run in train or application mode.')
    #parser.add_argument('-use_word_embeddings', dest='use_word_embeddings', required=True,
    #                    help='State whether word embeddings should be used as input.')
    parser.add_argument('-word_embeddings_src_model', dest='word_embeddings_src_save_path', required=False,
                        help='The path to the pretrained model with the word embeddings (for the source language).')
    parser.add_argument('-word_embeddings_src_train_data', dest='word_embeddings_src_train_data', required=False,
                        help='The path to the corpus used for training the word embeddings for the source language.')
    parser.add_argument('-word_embeddings_target_model', dest='word_embeddings_target_save_path', required=False,
                        help='The path to the pretrained model with the word embeddings (for the target language).')
    parser.add_argument('-word_embeddings_target_train_data', dest='word_embeddings_target_train_data', required=False,
                        help='The path to the corpus used for training the word embeddings for the target language.')
    parser.add_argument('-word_embedding_size', dest='word_embedding_size', required=True,
                        help='Size of the word embedding vectors.')
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=0.3,
                        help='How fast should the network learn.')
    parser.add_argument('-training_iterations', dest='training_iters', required=False, default=10000,
                        help='How many iterations should the network train for.')
    parser.add_argument('-batch_size', dest='batch_size', required=False, default=128,
                        help='Size of the training batches.')
    parser.add_argument('-n_hidden', dest='n_hidden', required=False, default=100,
                        help='Size of the hidden layer.')
    parser.add_argument('-sequence_width', dest='seq_width', required=False, default=50,
                        help='Maximum length of a sentence to be passed to the network (the rest is cut off).')
    parser.add_argument('-data', dest='data', required=False, default="brown",
                        help='The path to the gold corpus used for training/testing OR to the data to be processed.')
    parser.add_argument('-save_path', dest='save_path', required=False, default="None",
                        help='Path to where the model should be saved.')

    args = parser.parse_args()

    word_embedding_size = int(args.word_embedding_size)

    if args.train=="True":
        src2id = {} # map word strings to integers
        src_embeddings = {} # store the normalized embeddings; keys are integers (0 to n)
        with tf.Graph().as_default(), tf.Session() as session:
            opts = w2v.Options()
            opts.train_data = args.word_embeddings_src_train_data
            opts.save_path = args.word_embeddings_src_save_path
            opts.emb_dim = word_embedding_size
            model = w2v.Word2Vec(opts, session)
            ckpt = tf.train.get_checkpoint_state(args.word_embeddings_src_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("No valid checkpoint to reload a model was found!")
            src2id = model._word2id
            id2src = model._id2word
            src_embeddings = session.run(model._w_in)
            src_embeddings = tf.nn.l2_normalize(src_embeddings, 1).eval()

        target2id = {}
        target_embeddings = {}
        with tf.Graph().as_default(), tf.Session() as session:
            opts = w2v.Options()
            opts.train_data = args.word_embeddings_target_train_data
            opts.save_path = args.word_embeddings_target_save_path
            opts.emb_dim = word_embedding_size
            model = w2v.Word2Vec(opts, session)
            ckpt = tf.train.get_checkpoint_state(args.word_embeddings_target_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                model.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print("No valid checkpoint to reload a model was found!")
            target2id = model._word2id
            id2target = model._id2word
            target_embeddings = session.run(model._w_in)
            target_embeddings = tf.nn.l2_normalize(target_embeddings, 1).eval()
    else:
        src2id = pickle.load(open(os.path.join(args.save_path, "src2id.pkl"), "rb"))
        id2src = pickle.load(open(os.path.join(args.save_path, "id2src.pkl"), "rb"))
        target2id = pickle.load(open(os.path.join(args.save_path, "target2id.pkl"), "rb"))
        id2target = pickle.load(open(os.path.join(args.save_path, "id2target.pkl"), "rb"))

    # Network Parameters
    if args.train=="True":
        learning_rate = float(args.learning_rate) # Update rate for the weights
        training_iters = int(args.training_iters) # Number of training steps
        batch_size = int(args.batch_size) # Number of sentences passed to the network in one batch
    seq_width = int(args.seq_width) # Max sentence length (longer sentences are cut to this length)
    n_hidden = int(args.n_hidden) # Number of features/neurons in the hidden layer
    embedding_size = word_embedding_size
    source_vocab_size = len(src2id)
    target_vocab_size = len(target2id)
    if embedding_size == 0:
        print "No embedding model given as parameter!"
        exit(1)

    data = args.data
    # Format data
    if args.train=="True":
        # Get the training/validation/test data
        data = data_reader.get_data(data)
        random.shuffle(data)
        # number of sentences in the data is 727020
        valid_data_list = data[:500]
        test_data_list = data[500:1000]
        training_data_list = data[1000:]

        valid_input, valid_labels, valid_seq_length = format_data.format_data(valid_data_list, seq_width, src2id, target2id)
        test_input, test_labels, test_seq_length = format_data.format_data(valid_data_list, seq_width, src2id, target2id)
    else:
        app_data_list = data_reader.get_data_app(data)

    #TODO: Get a dictionary from source language lemmas to target language lemmas
    #src2target = dictionary_builder.get_alignment_dictionaty(data, src2id, target2id)
    #len = 0
    #for key, value in src2target:
    #    if len(value) > len:
    #        len = len(value)
    #print "Longest length of aligned word is: " + str(len)

    graph = tf.Graph()
    with graph.as_default():
        # Initialize the embedding layer, initially with a placeholder.
        # Later load pretrained embeddings: sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
        #W = tf.Variable(tf.zeros(shape=[len(src2id), word_embedding_size], dtype="float32"),
        #                trainable=True, name="W")
        if args.train=="True":
            W = tf.get_variable(name="W", shape=[len(src2id), word_embedding_size], initializer=tf.constant_initializer(src_embeddings),
                                trainable=True)
        else:
            W = tf.get_variable(name="W", shape=[len(src2id), word_embedding_size], trainable=True)
        #embedding_placeholder = tf.placeholder(tf.float32, [len(src2id), word_embedding_size])
        #embedding_init = W.assign(embedding_placeholder)

        if (args.train=="True"):
            tf_train_dataset = tf.placeholder(tf.int64, [batch_size, seq_width])
            tf_train_labels = tf.placeholder(tf.int64, [batch_size, seq_width])
            tf_train_seq_length = tf.placeholder(tf.int64, [batch_size])
            tf_valid_dataset = tf.constant(valid_input, tf.int64)
            tf_valid_labels = tf.constant(valid_labels, tf.int64)
            tf_valid_seq_length = tf.constant(valid_seq_length, tf.int64)
            tf_test_dataset = tf.constant(test_input, tf.int64)
            tf_test_labels = tf.constant(test_labels, tf.int64)
            tf_test_seq_length = tf.constant(test_seq_length, tf.int64)

            valid_labels_reshaped = tf.reshape(tf.transpose(valid_labels, [1, 0]), [-1, 1])
            test_labels_reshaped = tf.reshape(tf.transpose(tf_test_labels, [1, 0]), [-1, 1])

            train_embeddings = tf.nn.embedding_lookup(W, tf_train_dataset)
            valid_embeddings = tf.nn.embedding_lookup(W, tf_test_dataset)
            test_embeddings = tf.nn.embedding_lookup(W, tf_valid_dataset)
        else:
            inputs = tf.placeholder(tf.int64, [1,seq_width])
            input_embeddings = tf.nn.embedding_lookup(W, inputs)
            input_length = tf.placeholder(tf.int64)

        # Define weight matrices
        #TODO: Define matrices for each word in the source vocabulary
        #weights = {}
        #biases = {}
        #for word, id in src2id:
        #    weights[id] = tf.Variable(tf.random_normal([2*n_hidden, len(src2target[id])]))
        #    biases[id] = tf.Variable(tf.random_normal([len(src2target[id])]))
        #weights = {
        #    'out': tf.Variable(tf.random_normal([2*n_hidden, target_vocab_size]))
        #}
        #biases = {
        #   'out': tf.Variable(tf.random_normal([target_vocab_size]))
        #}

        # Bidirectional recurrent neural network with LSTM cells
        def BiRNN (inputs, _seq_length, compute_logits=False):

            # input shape: (batch_size, seq_width, embedding_size) ==> (seq_width, batch_size, embedding_size)
            inputs = tf.transpose(inputs, [1, 0, 2])
            # Reshape before feeding to hidden activation layers
            inputs = tf.reshape(inputs, [-1, embedding_size])
            # Split the inputs to make a list of inputs for the rnn
            inputs = tf.split(0, seq_width, inputs) # seq_width * (batch_size, n_hidden)
            initializer = tf.random_uniform_initializer(-1,1)
            with tf.variable_scope('forward'):
                #TODO: Use state_is_tuple=True
                fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, initializer=initializer)
            with tf.variable_scope('backward'):
                #TODO: Use state_is_tuple=True
                bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, initializer=initializer)
            # Get lstm cell output
            #tf.nn.bidirectional_rnn()
            outputs,_,_ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, inputs, dtype="float32", sequence_length=_seq_length)
            outputs_tensor = tf.reshape(tf.concat(0, outputs),[-1, 2*n_hidden])

            logits = []

            #TODO: Calculate logits separately for each word
            #for i in xrange(len(outputs)):
            #    logits_sent = []
            #   for j in xrange(len(outputs[i])):
            #       input_word_id = training_data_list[i][j]
            #        #TODO: Pad all word logits?
            #        logits_word = tf.matmul(outputs[i][j],weights[input_word_id]) + biases[input_word_id]
            #        logits_sent.append(logits_word)
            #   logits.append(logits_sent)

            # Compute logits across the whole vocabulary (slow for large dictionaries)
            if compute_logits==True:
                for i in xrange(len(outputs)):
                    final_transformed_val = tf.matmul(outputs[i],w) + b
                    logits.append(final_transformed_val)
                logits = tf.reshape(tf.concat(0, logits), [-1, target_vocab_size])

                return logits, outputs_tensor
            else:
                return outputs_tensor

        def calculate_logits (inputs):
            logits = []
            for i in xrange(len(inputs)):
                final_transformed_val = tf.matmul(inputs[i],w) + b
                logits.append(final_transformed_val)
            logits = tf.reshape(tf.concat(0, logits), [-1, target_vocab_size])

            return logits

        w = tf.get_variable("proj_w", [2*n_hidden, target_vocab_size])
        b = tf.get_variable("proj_b", [target_vocab_size])

        with tf.variable_scope("BiRNN") as scope:
            if args.train=="True":
                _outputs_tensor = BiRNN(train_embeddings, tf_train_seq_length)
                w = tf.get_variable("proj_w", [2*n_hidden, target_vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable("proj_b", [target_vocab_size])
                scope.reuse_variables()
                #TODO: In order to use this function, possibly pad all logit vectors to an equal length?
                #loss = tf.reduce_mean(
                #  tf.nn.softmax_cross_entropy_with_logits(
                #    logits, tf.reshape(tf.transpose(tf_train_gold, [1,0,2]), [-1, target_vocab_size])))
                # Define the weights for the hidden layer before the softmax (logits)
                train_gold_reshaped = tf.reshape(tf.transpose(tf_train_labels, [1,0]), [-1, 1])
                logits, labels = tf.nn._compute_sampled_logits(w_t, b, _outputs_tensor, train_gold_reshaped, 200,
                                                               target_vocab_size)
                #tf.reshape(tf.transpose(tf_train_labels, [1,0,2]), [-1, n_classes]))
                sampled_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
                #loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(w_t, b, tf_train_dataset, tf_train_gold, 200,
                #                                target_vocab_size))

                # try out a decaying learning rate
                #global_step = tf.Variable(0)  # count the number of steps taken.
                #learning_rate = tf.train.exponential_decay(learning_rate, global_step, 3500, 0.86, staircase=True)

                # calculate gradients, clip them and update model in separate steps
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                gradients = optimizer.compute_gradients(sampled_loss)
                capped_gradients = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gradients if grad!=None]
                optimizer_t = optimizer.apply_gradients(capped_gradients)

                # Predictions for the training, validation, and test data.
                #train_prediction = tf.nn.softmax(logits)
                #valid_prediction = tf.nn.softmax(BiRNN(valid_embeddings, tf_valid_seq_length)[0])
                #test_prediction = tf.nn.softmax(BiRNN(test_embeddings, tf_test_seq_length)[0])
                train_prediction = tf.nn.softmax(logits)
                valid_outputs = BiRNN(valid_embeddings, tf_valid_seq_length)
                valid_logits_sampled, valid_labels_sampled = tf.nn._compute_sampled_logits(w_t, b, valid_outputs, valid_labels_reshaped,
                                                                               200, target_vocab_size)
                valid_prediction_sampled = tf.nn.softmax(valid_logits_sampled)
                test_outputs = BiRNN(test_embeddings, tf_test_seq_length)
                test_logits_sampled, test_labels_sampled = tf.nn._compute_sampled_logits(w_t, b, valid_outputs, test_labels_reshaped,
                                                                               200, target_vocab_size)
                test_prediction_sampled = tf.nn.softmax(test_logits_sampled)
            else:
                _logits = BiRNN(input_embeddings, input_length, compute_logits=True)[0]
                predictions = tf.nn.softmax(_logits)

    # Create a new batch from the training data (data, labels and sequence lengths)
    def new_batch (offset, data_list):

        batch = training_data_list[offset:(offset+batch_size)]
        train_input, train_labels, seq_length = format_data.format_data(batch, seq_width, src2id, target2id)
        return train_input, train_labels, seq_length

    # Function to calculate the accuracy on a batch of results and gold labels '''
    def accuracy (predictions, labels):

        #reshaped_labels = np.reshape(np.transpose(labels, (1,0)), (-1,1))
        matching_cases = 0
        eval_cases = 0
        # Do not count results beyond the end of a sentence (in the case of sentences shorter than 50 words)
        for i in xrange(predictions.shape[0]):
            # If all values in a gold POS label are zeros, skip this calculation
            if max(labels[i]) == 0:
                continue
            if np.argmax(labels[i]) == np.argmax(predictions[i]):
                matching_cases+=1
            eval_cases+=1
        return (100.0 * matching_cases) / eval_cases

    # Run the tensorflow graph
    with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        if args.train=="True":
            tf.initialize_all_variables().run()
            # Load pretrained embeddings
            #session.run(embedding_init, feed_dict={embedding_placeholder: src_embeddings})
            print('Initialized')
            for step in range(training_iters):
                offset = (step * batch_size) % (len(training_data_list) - batch_size)
                batch_input, batch_labels, batch_seq_length = new_batch(offset, training_data_list)
                feed_dict = {tf_train_dataset : batch_input, tf_train_labels : batch_labels, tf_train_seq_length: batch_seq_length}
                _, l, predictions, _labels = session.run(
                  [optimizer_t, sampled_loss, train_prediction, labels], feed_dict=feed_dict)
                if (step % 50 == 0):
                  print 'Minibatch loss at step ' + str(step) + ': ' + str(l)
                  print 'Minibatch accuracy: ' + str(accuracy(predictions, _labels))
                  print 'Validation accuracy: ' + str(accuracy(valid_prediction_sampled.eval(), valid_labels_sampled.eval()))
                  if (args.save_path != "None"):
                    saver.save(session, os.path.join(args.save_path, "model.ckpt"), global_step=step)
                    with open(os.path.join(args.save_path, 'src2id.pkl'), 'wb') as output:
                        pickle.dump(src2id, output, pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(args.save_path, 'id2src.pkl'), 'wb') as output:
                        pickle.dump(id2src, output, pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(args.save_path, 'target2id.pkl'), 'wb') as output:
                        pickle.dump(target2id, output, pickle.HIGHEST_PROTOCOL)
                    with open(os.path.join(args.save_path, 'id2target.pkl'), 'wb') as output:
                        pickle.dump(id2target, output, pickle.HIGHEST_PROTOCOL)


            print 'Test accuracy: ' + str(accuracy(test_prediction_sampled.eval(), test_labels_sampled.eval()))
            #if (args.save_path != "None"):
            #    model.saver.save(session, os.path.join(args.save_path, "model.ckpt"), global_step=model.step)
        else:
            saver.restore(session, os.path.join(args.save_path, "model.ckpt-3000"))
            sents = []
            _inputs, _seq_length = format_data.format_data_app(app_data_list, seq_width, src2id)
            for i in range(_inputs.shape[0]):
                print "Input sentence is: ",
                for j in xrange(len(app_data_list[i])):
                    print app_data_list[i][j] + " ",
                print "\n"
                feed_dict = {inputs: [_inputs[i]], input_length: [_seq_length[i]]}
                _predictions = session.run([predictions], feed_dict=feed_dict)[0]
                #_predictions = _predictions.eval()
                print "Output sequence is: ",
                for k in xrange(_predictions.shape[0]):
                    print id2target[np.argmax(_predictions[k])] + " ",
                print "\n"

