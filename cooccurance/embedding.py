import numpy as np
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD,KernelPCA, PCA, SparsePCA

co_occ = np.load('co_occ_char.npy')
print(co_occ.shape)
print('pcaing...')
kpca = KernelPCA(n_components = 300, kernel='rbf')
co_occ_kpca = kpca.fit_transform(co_occ)
print(co_occ_kpca.shape)



unigram_ranked_list_of_words = []
f1 = open("Unigrams_char.txt", "r")
for line in tqdm(f1):
    line = line.rstrip()
    unigram = (line.split(":")[0]).rstrip()
    unigram_ranked_list_of_words.append(unigram)
wordlist = np.array(unigram_ranked_list_of_words)

wordlen = len(unigram_ranked_list_of_words)
wordlist = wordlist.reshape((wordlen,1))

result = np.concatenate((wordlist, co_occ_kpca),axis=1)
print(result.shape)

np.save('result_char_rdf.npy', result)