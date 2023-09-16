import numpy as np
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import treebank
from tqdm import tqdm

nltk.download('treebank')
nltk.download('punkt')

class POSTagger:
    def __init__(self) -> None:
        self.tag_set = set(tag for _, tag in treebank.tagged_words())
        self.tag2id = {tag:i for i, tag in enumerate(self.tag_set)}
        self.id2tag = {i:tag for i, tag in enumerate(self.tag_set)}
        self.word2id = {word:i for i, word in enumerate(treebank.words())}
        self.tokenizer = TreebankWordTokenizer()

        self.start_prob = np.zeros(len(self.tag_set), dtype=np.float32)
        self.transition_prob = np.zeros((len(self.tag_set), len(self.tag_set)), dtype=np.float32)
        self.emission_prob = np.zeros((len(self.tag_set), len(treebank.words())), dtype=np.float32)

    def maptag2id(self, tag: str):
        return self.tag2id[tag]

    def mapword2id(self, word: str):
        if word in self.word2id:
            return self.word2id[word]
        return None

    def train(self):
        print("Computing probabilities using MLE ...")
        for sent in tqdm(treebank.tagged_sents()):
            tag_id = self.maptag2id(sent[0][1])
            self.start_prob[tag_id] += 1
            for i in range(len(sent)-1):
                self.transition_prob[self.maptag2id(sent[i][1])][self.maptag2id(sent[i+1][1])] += 1
            for word, tag in sent:
                self.emission_prob[self.maptag2id(tag)][self.mapword2id(word)] += 1

        #normalize probabilities
        print("Normalizing probabilities...")
        self.start_prob /= len(treebank.tagged_sents())
        self.transition_prob /= np.sum(self.transition_prob, axis=1, keepdims=True)
        self.emission_prob /= np.sum(self.emission_prob, axis=1, keepdims=True)
        
    #implement the Viterbi algorithm
    def viterbi(self, sentence):

        if type(sentence) == str:
            tokens = self.tokenizer.tokenize(sentence)
        elif type(sentence) == list:
            tokens = sentence
        prob_table = np.zeros((len(tokens), len(self.tag_set)), dtype=np.float32)
        back = np.zeros((len(tokens), len(self.tag_set)), dtype=np.int32)

        #initialization
        for t in range(len(tokens)):
            for s in range(len(self.tag_set)):
                if t==0:
                    if self.mapword2id(tokens[0]) == None: # if the word is not in the corpus
                        em_prob = 1 
                    else:
                        em_prob = self.emission_prob[s][self.mapword2id(tokens[0])]                    
                    prob_table[0][s] = self.start_prob[s] * em_prob
                else:
                    prob_list = []
                    for i in range(len(self.tag_set)):
                        if self.mapword2id(tokens[t]) == None: # if the word is not in the corpus   
                            em_prob = 1
                        else:
                            em_prob = self.emission_prob[s][self.mapword2id(tokens[t])]
                        prob_list.append(prob_table[t-1][i] * self.transition_prob[i][s] * em_prob)

                    prob_table[t][s] = np.max(prob_list)
                    back[t][s] = np.argmax(prob_list)

        final_tag = np.argmax(prob_table[-1])
        tag_seq = []
        tag_seq.append(final_tag)

        # backtracking
        for t in range(len(tokens)-1, 0, -1):
            tag_seq.insert(0, back[t][tag_seq[0]])

        tag_seq = list(map(lambda x: self.id2tag[x], tag_seq))
        return list(zip(tokens, tag_seq))
