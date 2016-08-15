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
