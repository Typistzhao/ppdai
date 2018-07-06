import re
import operator


class Unigram:
    def __init__(self):
        self.f1 = None  # sample file to read
        self.inputpath = None
        self.unigrams_dict = {}  # Dictionary to hold unigrams and its count
        self.dict_plot = {}  # x=key=rank,y=value=frequency
        self.ranked_list = []  # unigrams
        self.unigram_feature_vector = {}
        self.unigram_ranked_list_of_words = []
        self.outputpath = None
        self.no_of_unigrams = 0

    def set_input_path(self, inputpath):
        self.inputpath = inputpath

    def set_output_path(self, outputpath):
        self.outputpath = outputpath

    def add_unigram_to_list(self, token, main_word):  # unigrams_dict is of the form {token:[frequency,case satisfied,last word from which token generated]}
        if token not in self.unigrams_dict:
            self.unigrams_dict[token] = [1, main_word]
            self.no_of_unigrams = self.no_of_unigrams + 1
        else:
            count = self.unigrams_dict[token][0] + 1
            self.unigrams_dict[token] = [count, main_word]

    def get_no_of_unigrams(self):
        return self.no_of_unigrams

    def find_grams(self, token):
        self.add_unigram_to_list(token, token)

    def sort_descending(self, unigrams):
        return sorted(unigrams.items(), key=operator.itemgetter(1), reverse=True)

    def find_x_y_for_plotting(self, ranked_list, grams_dict, filepath):
        dict_plot = {}
        f1 = open(filepath, "w")
        for i in range(len(ranked_list)):
            word = ranked_list[i][0]  # fetching the word string from ranked list
            dict_plot[i + 1] = grams_dict[word][0]  # finding the frequency of word from unigrams_dict
            f1.write(word + "\t\t:\t" + str(grams_dict[word][0]) + "\n")

        f1.close()
        return dict_plot

    def find_unigram(self):
        self.f1 = open(self.inputpath, "r")
        i = 1
        for line in self.f1:
            self.last_token = ""
            if i % 1000 == 0:
                print(i)

            i = i + 1
            tokens = line.rstrip().split()
            for token in tokens:
                self.find_grams(token)

        self.ranked_list = self.sort_descending(self.unigrams_dict)  # key=word,value=frequency
        unigram_output_path = self.outputpath
        self.find_x_y_for_plotting(self.ranked_list, self.unigrams_dict, unigram_output_path)
        print("self.no_of_unigrams:" + str(self.no_of_unigrams))



def main():
    u = Unigram()
    u.set_input_path("char_doc.txt")
    u.set_output_path("Unigrams_char.txt")
    u.find_unigram()


if __name__ == '__main__':
    main()