def get_data(path):
    with open(path, "r") as corpus:
        pairs = corpus.read().split("@@@")
        training_data = []
        for pair in pairs:
            try:
                source, target = pair.strip().split("\n")
            except:
                print "Couldn't unpack sentences: " + pair.strip()
            src_words = source.split(" ")
            tgt_words = target.split(" ")
            try:
                data_point = zip(src_words, tgt_words)
                training_data.append(data_point)
            except:
                print "Couldn't merge source and target sentences.\n" + "Source: " + source +\
                      "\nTarget: " + target
    return training_data

def get_data_app(path):
    with open(path, "r") as corpus:
        sents = corpus.read().split("@@@")
        data = []
        for sent in sents:
            words = sent.split(" ")
            data.append(words)
    return data