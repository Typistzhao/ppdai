'''训练集长度'''
train_len = train[['q1','q2']].applymap(lambda x: len(x.split()))


'''测试集长度'''
test_len = test[['q1','q2']].applymap(lambda x: len(x.split()))


'''公共个数'''
def num_of_common(q1,q2):
    t1 = np.asarray(re.split(' ',q1))
    t2 = np.asarray(re.split(' ',q2))
    return len(np.intersect1d(t1,t2))
common_num = train[['q1','q2']].apply(lambda x: num_of_common(x['q1'],x['q2']),axis=1)


'''最长公共子序列'''
def lcs_length(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]
lcs_len = train[['q1','q2']].apply(lambda x: lcs_length(x['q1'].split(),x['q2'].split()),axis=1)



'''编辑距离'''
def levenshtein(q1, q2):
    len_q1 = len(q1) + 1
    len_q2 = len(q2) + 1
    #create matrix
    matrix = [0 for n in range(len_q1 * len_q2)]
    #init x axis
    for i in range(len_q1):
        matrix[i] = i
    #init y axis
    for j in range(0, len(matrix), len_q1):
        if j % len_q1 == 0:
            matrix[j] = j // len_q1
    for i in range(1, len_q1):
        for j in range(1, len_q2):
            if q1[i-1] == q2[j-1]:
                cost = 0
            else:
                cost = 1
            matrix[j*len_q1+i] = min(matrix[(j-1)*len_q1+i]+1,
                    matrix[j*len_q1+(i-1)]+1,
                    matrix[(j-1)*len_q1+(i-1)] + cost)
    return matrix[-1]
edit_dis = train[['q1','q2']].apply(lambda x: levenshtein(x['q1'].split(),x['q2'].split()),axis=1)