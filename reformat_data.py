import re, copy
from collections import Counter

f_in = "/home/alexander/dev/projects/BAN/lexical-language-model/data/bg.A3.final"
#f_in = "/home/alexander/dev/projects/BAN/lexical-language-model/data/test.txt"
#f_in = "/home/alexander/dev/projects/BAN/lexical-language-model/data/mini_test.txt"
f_new = "/home/alexander/dev/projects/BAN/lexical-language-model/data/en-bg_new.txt"
#f_new = "/home/alexander/dev/projects/BAN/lexical-language-model/data/new_test.txt"
#f_new = "/home/alexander/dev/projects/BAN/lexical-language-model/data/new_mini_test.txt"
f_stopwords_bg = "/home/alexander/dev/projects/BAN/lexical-language-model/data/stopwords/stopwords_bg_utf8.txt"
f_stopwords_en = "/home/alexander/dev/projects/BAN/lexical-language-model/data/stopwords/stopwords_en.txt"

with open(f_in, "r") as input_data, open(f_stopwords_bg, "r") as stop_bg, open(f_stopwords_en, "r") as stop_en:
    #get stopword lists for src and trgt languages
    stopwords_bg = stop_bg.readlines()
    stopwords_bg = map(str.strip, stopwords_bg)
    stopwords_en = stop_en.readlines()
    stopwords_en = map(str.rstrip, stopwords_en)

    pairs = input_data.read().split("# Sentence pair")
    paired_sents = []
    for pair in pairs:
        lines = pair.strip().split("\n")
        '''
        print lines
        for line in lines:
            print line
            print "***"
        '''
        if len(lines) < 3:
            continue
        source, target = lines[1:]
        #print source, "BREAK", target
        #print "****"
        source_words = source.split()
        target = re.sub(r'NULL \(\{ [0-9 ]+ \}\)', '', target)
        #print target
        target_mappings = target.split("})")
        src_to_target = []
        seen_pointers = {}
        i = 0
        for mapping in target_mappings:
            if "({" not in mapping:
                continue
            word, pointers = mapping.split(" ({ ")
            word = word.strip()
            #if word in stopwords_bg:
            #    continue
            pointers = pointers.strip()
            if pointers == "":
                continue
            pointers = map(int, pointers.split(" "))
            for pointer in pointers:
                if len(pointers) > 1 and source_words[pointer-1] in stopwords_en:
                    continue
                src_to_target.append((source_words[pointer-1], word))
                if pointer not in seen_pointers:
                    seen_pointers[pointer] = [(i, word)]
                else:
                    seen_pointers[pointer].append((i, word))
                i += 1
        #print src_to_target

        popped_words = 0
        for pointer, words in seen_pointers.iteritems():
            #print words
            if len(words) > 1:
                remaining_words = []
                for word in words:
                    if word[1] in stopwords_bg:
                        src_to_target.pop(word[0]-popped_words)
                        popped_words+=1
                        continue
                    remaining_words.append(word)
                remaining_words = sorted(remaining_words)
                if len(remaining_words) > 1:
                    for word in remaining_words[1:]:
                        src_to_target.pop(word[0]-popped_words)
                        popped_words+=1

        paired_sents.append(src_to_target)
    #print paired_sents

with open(f_new, "w") as new_data:
    for pair_sents in paired_sents:
        src_sent = ""
        trgt_sent = ""
        for pair in pair_sents:
            src_sent += pair[0] + " "
            trgt_sent += pair[1] + " "
        src_sent = src_sent.rstrip()+"\n"
        trgt_sent = trgt_sent.rstrip() + "\n"
        new_data.write(src_sent)
        new_data.write(trgt_sent)
        new_data.write("@@@\n")
