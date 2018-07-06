import numpy as np
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD,pca,KernelPCA

unigram_path = "Unigrams_char.txt"
bigram_path = "Bigrams_char.txt"


unigram_ranked_list_of_words = []
f1 = open(unigram_path, "r")
for line in f1:
    line = line.rstrip()
    unigram = (line.split(":")[0]).rstrip()
    unigram_ranked_list_of_words.append(unigram)
wordlist = np.array(unigram_ranked_list_of_words)

wordlen = len(unigram_ranked_list_of_words)
wordlist = wordlist.reshape((wordlen,1))

bigram_ranked_dict = {}
f2 = open(bigram_path, "r")
for line in f2:
    line = line.strip()
    bigram_tokens = line.split("\t\t:\t")
    bigram = bigram_tokens[0]
    count = bigram_tokens[1]
    bigram_ranked_dict[bigram] = int(count)

def neighbour_count(word1, word2):
    bigram1 = "(" + word1 + "," + word2 + ")"
    bigram2 = "(" + word2 + "," + word1 + ")"
    count = 0
    if bigram1 in bigram_ranked_dict:
        count = count + bigram_ranked_dict[bigram1]
    if bigram2 in bigram_ranked_dict:
        count = count + bigram_ranked_dict[bigram2]
    return count


# co_occ = numpy.memmap("co_occ", dtype='float32', mode='w+', shape=(wordlen, wordlen))
co_occ = np.zeros((wordlen, wordlen))
for i in tqdm(range(wordlen)):
    # if i % 1000 == 0:
    #     print(i)
    wordi = unigram_ranked_list_of_words[i]
    for j in range(wordlen):
        wordj = unigram_ranked_list_of_words[j]
        feature_value = neighbour_count(wordi, wordj)
        co_occ[i, j] = feature_value
np.save('co_occ_char.npy', co_occ)
# np.savetxt('co_occ.txt', co_occ)

# svd = TruncatedSVD(n_components = 300)
# co_occ_svd = svd.fit_transform(co_occ)

# co_occ = np.loadtxt('co_occ.txt')

# svd = KernelPCA(n_components = 300, kernel = 'cosine')
# co_occ_svd = svd.fit_transform(co_occ)
#
#
#
# result = np.column_stack((wordlist,co_occ_svd))