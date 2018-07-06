# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 00:30:12 2015
@author: nausheenfatma
"""

import numpy
from numpy import linalg as la


class WordCoOccuranceMatrix:

    def __init__(self):
        self.feature_word_index = {}
        self.co_occ = None
        self.feature_vector_size = 0
        self.bigram_ranked_dict = {}
        self.unigram_ranked_list_of_words = []
        self.unigram_path = "Unigrams.txt"
        self.bigram_path = "Bigrams.txt"
        self.load_properties()

    def load_properties(self):
        self.co_occ_path = "cooccuranceMatrix.txt"
        self.feature_vector_size = 20890

        print("elf.feature_vector_size", self.feature_vector_size)


    def makefeaturevector(self):
        self.unigram_ranked_list_of_words = self.fetch_ranked_unigrams()
        self.set_index_for_feature()

    def fetch_ranked_unigrams(self):
        print("self.unigram_path", self.unigram_path)

        f1 = open(self.unigram_path, "r")
        unigram_list = []
        for line in f1:
            line = line.rstrip()
            unigram = (line.split(":")[0]).rstrip()
            unigram_list.append(unigram)
        #   print "unigram_list",unigram_list
        return unigram_list

    def fetch_bigrams(self):
        f1 = open(self.bigram_path, "r")
        for line in f1:
            line = line.strip()
            bigram_tokens = line.split("\t\t:\t")
            bigram = bigram_tokens[0]
            count = bigram_tokens[1]
            self.bigram_ranked_dict[bigram] = int(count)

    def make_cooccurence_matrix(self):
        print(len(self.feature_word_index))

        file1 = open("wordlist.txt", "w")
        print("len(self.unigram_ranked_list_of_words)", len(self.unigram_ranked_list_of_words))

        self.co_occ = numpy.memmap(self.co_occ_path, dtype='float32', mode='w+',
                                   shape=(len(self.unigram_ranked_list_of_words), self.feature_vector_size))
        for i in range(len(self.unigram_ranked_list_of_words)):
            if i % 10 == 0:
                print(i)

            word = self.unigram_ranked_list_of_words[i]
            file1.write(word)
            for j in range(len(self.feature_word_index)):
                feature_index = j
                feature_value = self.neighbour_count(word, self.feature_word_index[feature_index])
                self.co_occ[i, j] = feature_value
                # file1.write(str(feature_value) + "\t")
            file1.write("\n")
        self.co_occ.flush()

    def set_index_for_feature(self):
        print("len(self.unigram_ranked_list_of_words)", len(self.unigram_ranked_list_of_words))

        print(self.unigram_ranked_list_of_words[1])

        print("self.feature_size", self.feature_vector_size)

        for i in range(self.feature_vector_size):
            self.feature_word_index[i] = self.unigram_ranked_list_of_words[i]

    def neighbour_count(self, word1, word2):

        bigram1 = "(" + word1 + "," + word2 + ")"
        bigram2 = "(" + word2 + "," + word1 + ")"
        count = 0
        if bigram1 in self.bigram_ranked_dict:
            count = count + self.bigram_ranked_dict[bigram1]
        if bigram2 in self.bigram_ranked_dict:
            count = count + self.bigram_ranked_dict[bigram2]
        return count

    def svd(self):
        U, sigma, VT = la.svd(self.co_occ)

    def find_co_occurance_matrix(self):
        self.makefeaturevector()
        self.fetch_bigrams()
        self.make_cooccurence_matrix()


def main():
    wcm = WordCoOccuranceMatrix()
    wcm.find_co_occurance_matrix()


if __name__ == '__main__':
    main()