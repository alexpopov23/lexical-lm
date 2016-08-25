import argparse
import random
import os
import pickle

import tensorflow as tf
import numpy as np

import word2vec_optimized_utf8 as w2v
import data_reader
import format_data
import dictionary_builder

if __name__ == "__main__":

    parser = argparse.ArgumentParser(version='1.0',description='Train a neural POS tagger.')
    parser.add_argument('-word_embeddings_src_model', dest='word_embeddings_src_save_path', required=False,
                        help='The path to the pretrained model with the word embeddings (for the source language).')
    parser.add_argument('-word_embeddings_src_train_data', dest='word_embeddings_src_train_data', required=False,
                        help='The path to the corpus used for training the word embeddings for the source language.')
    parser.add_argument('-word_embedding_size', dest='word_embedding_size', required=False,
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
    parser.add_argument('-training_data', dest='training_data', required=True, default="brown",
                        help='The path to the gold corpus used for training/testing.')
    parser.add_argument('-save_path', dest='save_path', required=False, default="None",
                        help='Path to where the model should be saved.')

    args = parser.parse_args()

    word_embedding_size = int(args.word_embedding_size)

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

    target2id, id2target = dictionary_builder.get_target_dict(args.training_data)

    # Network Parameters
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

    data = args.training_data
    # Get the training/validation/test data
    data = data_reader.get_data(data)
    random.shuffle(data)
    training_data_list = data

    graph = tf.Graph()
    with graph.as_default():
        W = tf.get_variable(name="W", shape=[len(src2id), word_embedding_size], initializer=tf.constant_initializer(src_embeddings),
                            trainable=True)
        tf_train_dataset = tf.placeholder(tf.int64, [batch_size, seq_width])
        tf_train_labels = tf.placeholder(tf.float32, [batch_size, seq_width, target_vocab_size])
        tf_train_seq_length = tf.placeholder(tf.int64, [batch_size])
        train_embeddings = tf.nn.embedding_lookup(W, tf_train_dataset)

        # Bidirectional recurrent neural network with LSTM cells
        def BiRNN (inputs, _seq_length):

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
            outputs,_,_ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, inputs, dtype="float32", sequence_length=_seq_length)
            outputs_tensor = tf.reshape(tf.concat(0, outputs),[-1, 2*n_hidden])

            logits = []
            for i in xrange(len(outputs)):
                final_transformed_val = tf.matmul(outputs[i],w) + b
                logits.append(final_transformed_val)
            logits = tf.reshape(tf.concat(0, logits), [-1, target_vocab_size])

            return logits, outputs_tensor

        w = tf.get_variable("proj_w", [2*n_hidden, target_vocab_size], initializer=tf.random_uniform_initializer(-1,1))
        b = tf.get_variable("proj_b", [target_vocab_size], initializer=tf.random_uniform_initializer(-1,1))

        with tf.variable_scope("BiRNN") as scope:
            logits, _outputs_tensor = BiRNN(train_embeddings, tf_train_seq_length)
            w = tf.get_variable("proj_w", [2*n_hidden, target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [target_vocab_size])
            scope.reuse_variables()
            train_gold_reshaped = tf.reshape(tf.transpose(tf_train_labels, [1,0,2]), [-1, target_vocab_size])
            #TODO Use sparse_softmax_cross_entropy_with_logits to save memory
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, train_gold_reshaped))
            # calculate gradients, clip them and update model in separate steps
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            gradients = optimizer.compute_gradients(loss)
            capped_gradients = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in gradients if grad!=None]
            optimizer_t = optimizer.apply_gradients(capped_gradients)
            train_prediction = tf.nn.softmax(logits)

    # Create a new batch from the training data (data, labels and sequence lengths)
    def new_batch (offset):

        batch = training_data_list[offset:(offset+batch_size)]
        train_input, train_labels, seq_length = format_data.format_data_fullsoftmax(batch, seq_width, src2id, target2id,
                                                                                    embedding_size)
        return train_input, train_labels, seq_length

    # Function to calculate the accuracy on a batch of results and gold labels '''
    def accuracy (predictions, labels, seq_lengths):

        reshaped_labels = np.reshape(np.transpose(labels, (1, 0, 2)), (-1, target_vocab_size))
        matching_cases = 0
        eval_cases = 0
        # Do not count results beyond the end of a sentence (in the case of sentences shorter than 50 words)
        for i in xrange(seq_width):
            for j in xrange(batch_size):
                if i+1 > seq_lengths[j]:
                    continue
                if np.argmax(reshaped_labels[i*batch_size + j]) == np.argmax(predictions[i*batch_size + j]):
                    matching_cases+=1
                eval_cases+=1
        return (100.0 * matching_cases) / eval_cases

    # Run the tensorflow graph
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        # Load pretrained embeddings
        print('Initialized')
        saver = tf.train.Saver()
        for step in range(training_iters):
            offset = (step * batch_size) % (len(training_data_list) - batch_size)
            batch_input, batch_labels, batch_seq_length = new_batch(offset)
            feed_dict = {tf_train_dataset : batch_input, tf_train_labels : batch_labels, tf_train_seq_length: batch_seq_length}
            _, l, predictions = session.run(
              [optimizer_t, loss, train_prediction], feed_dict=feed_dict)
            if (step % 100 == 0):
              print 'Minibatch loss at step ' + str(step) + ': ' + str(l)
              print 'Minibatch accuracy: ' + str(accuracy(predictions, batch_labels, batch_seq_length))
              if (args.save_path != "None" and step % 1000 == 0):
                saver.save(session, os.path.join(args.save_path, "model.ckpt"), global_step=step)
                with open(os.path.join(args.save_path, 'src2id.pkl'), 'wb') as output:
                    pickle.dump(src2id, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'id2src.pkl'), 'wb') as output:
                    pickle.dump(id2src, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'target2id.pkl'), 'wb') as output:
                    pickle.dump(target2id, output, pickle.HIGHEST_PROTOCOL)
                with open(os.path.join(args.save_path, 'id2target.pkl'), 'wb') as output:
                    pickle.dump(id2target, output, pickle.HIGHEST_PROTOCOL)
