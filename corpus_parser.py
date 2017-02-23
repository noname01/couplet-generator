import os
import numpy as np
import random
random.seed(10)

def load_corpus_a():
    prev = []
    corpus = []
    with open('corpus_a.txt', encoding='utf8') as fp:
        for line in fp:
            if line != '':
                tokens = [s.strip() for s in line.split(r' ') if s != '']
                if len(tokens) == len(prev) and len(tokens) > 0 and len(tokens[0]) == len(prev[0]):
                    for i in range(len(tokens)):
                        # print(prev[i] + ' ' + tokens[i])
                        corpus += [ (prev[i], tokens[i]) ]
                    prev = []
                else:
                    prev = tokens
    return corpus

def load_corpus_b(path='corpus_b.txt'):
    corpus = []
    with open(path, encoding='utf8') as fp:
        lines = [line.strip() for line in fp]
        for i in range(len(lines)):
            if lines[i] != '':
                tokens = [s for s in lines[i].split(r' ') if s != '']
                if len(tokens) == 1:
                    corpus += [ [tokens[0], lines[i + 1]] ]
                    lines[i + 1] = ''
                else:
                    corpus += [ (tokens[0], tokens[1]) ]
    return corpus

def load_corpus_c():
    # target = codecs.open('corpus_c.txt', 'w', 'utf-8')
    # with open('poetry.txt') as fp:
    #     for line in fp:
    #         if line != '':
    #             tokens = [s.strip() for s in line.decode('utf-8').strip().split(r' ') if s != '']
    #             if len(tokens) == 8 and len(tokens[2]) == len(tokens[3]) and len(tokens[4]) == len(tokens[5]):
    #                 line1 = tokens[2] + u' ' + tokens[3]
    #                 line2 = tokens[4] + u' ' + tokens[5]
    #                 target.write(line1 + '\n')
    #                 target.write(line2 + '\n')
    # target.close()
    return load_corpus_b('corpus_c.txt')

def line_to_ids(line):
    return np.array([char_to_int_map[char] for char in line])

def id_to_char(k):
    if (k < 0 or k >= len(all_chars)):
        print('k=' + str(k))
    assert k >= 0 and k < len(all_chars)
    return all_chars[k]

def vec_to_line(x):
    return ''.join([id_to_char(k) for k in x])

def get_corpus(covert_to_ints=True, seq_len=50):
    padded_pairs = [(pair[0] + '▁' * (seq_len - len(pair[0])),
        pair[1] + '▁' * (seq_len - len(pair[1]))) for pair in all_pairs if len(pair[0]) <= seq_len]
    if not covert_to_ints:
        return padded_pairs
    int_pairs = np.array([(line_to_ids(pair[0]), line_to_ids(pair[1])) for pair in padded_pairs])
    # print(int_pairs.shape)
    return int_pairs, len(all_chars)

ca = load_corpus_a()
cb = load_corpus_b()
cc = load_corpus_c()

all_pairs = ca
# all_pairs = cb + ca + cc
# print(len(all_pairs))

all_chars = set()
for pair in all_pairs:
    for line in pair:
        for char in line:
            all_chars.add(char)

all_chars = ['▁'] + sorted(list(all_chars))

char_to_int_map = {}
for i in range(len(all_chars)):
    char_to_int_map[all_chars[i]] = i

random.shuffle(all_pairs)

# print(len(all_chars))
# 5373

# max_len = np.max(np.array([len(pair[0]) for pair in all_pairs]))
# 44

# print(get_corpus(covert_to_ints=False)[0:10])

# print(vec_to_line(np.array([10, 20, 30] + [0] * 47)))