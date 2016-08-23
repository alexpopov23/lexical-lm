import bisect

def get_alignment_dictionaty(aligned_sents, src2id, target2id):

    dict = {}
    for alignment in aligned_sents:
        for pair in alignment:
            if src2id[pair[0]] not in dict:
                dict[src2id[pair[0]]] = [target2id[pair[1]]]
            else:
                bisect.insort(dict[src2id[pair[0]]], pair[1])
    return dict

def get_target_dict (data):

    target2id = {}
    target2id["UNK"] = 0
    word2count = {}
    with open(data,"r") as corpus:
        pairs = corpus.read().split("@@@")
        for pair in pairs:
            try:
                _, bg = pair.strip().split("\n")
            except:
                print "Couldn't unpack sentences: " + pair.strip()
                continue
            words_bg = bg.split(" ")
            for word in words_bg:
                if word in word2count:
                    word2count[word] += 1
                else:
                    word2count[word] = 1
    word2count_frequent = {word: word2count[word] for word in word2count.iterkeys() if word2count[word] > 5}
    sorted_words = sorted(word2count_frequent, key=word2count_frequent.get, reverse=True)
    count = 1
    for word in sorted_words:
        target2id[word] = count
        count += 1
    id2target = {v: k for k, v in target2id.items()}

    return target2id, id2target
