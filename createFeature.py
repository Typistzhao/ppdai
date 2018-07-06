import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import multiprocessing as mlp
from sklearn.decomposition import TruncatedSVD

tqdm.pandas()

def hash_q(train_orig,test_orig):


    df1 = train_orig[['q1']].copy()
    df2 = train_orig[['q2']].copy()
    df1_test = test_orig[['q1']].copy()
    df2_test = test_orig[['q2']].copy()

    df2.rename(columns={'q2': 'q1'}, inplace=True)
    df2_test.rename(columns={'q2': 'q1'}, inplace=True)

    train_questions = df1.append(df2)
    train_questions = train_questions.append(df1_test)
    train_questions = train_questions.append(df2_test)
    train_questions.drop_duplicates(subset=['q1'], inplace=True)

    train_questions.reset_index(inplace=True, drop=True)
    questions_dict = pd.Series(train_questions.index.values, index=train_questions.q1.values).to_dict()
    train_cp = train_orig.copy()
    test_cp = test_orig.copy()

    train_cp['label'] = 1
    test_cp['label'] = -1
    comb = pd.concat([train_cp, test_cp])

    comb['q1_hash'] = comb['q1'].map(questions_dict)
    comb['q2_hash'] = comb['q2'].map(questions_dict)

    train_comb = comb[comb['label'] >= 0][['q1_hash', 'q2_hash']]
    test_comb = comb[comb['label'] < 0][['q1_hash', 'q2_hash']]

    train_orig = pd.concat([train_orig,train_comb], axis=1)
    test_orig = pd.concat([test_orig,test_comb], axis=1)

    return train_orig,test_orig

def adj_feat_worker(data,FG,suffix):
    def get_weights_adj(x):
        q1 = x['q1']
        q2 = x['q2']
        q1_adj = set(FG[q1])
        q2_adj = set(FG[q2])

        q1_or_q2 = q1_adj | q2_adj
        total_weight = 0
        for node in q1_or_q2:
            if node in FG[q1]:
                total_weight+=FG.get_edge_data(q1,node)['weight']
            if node in FG[q2]:
                total_weight+=FG.get_edge_data(q2,node)['weight']
        x['q1q2_union'+suffix] = total_weight

        total_weight = 0
        q1_and_q2 = q1_adj & q2_adj
        for node in q1_and_q2:
            if node in FG[q1]:
                total_weight += FG.get_edge_data(q1, node)['weight']
            if node in FG[q2]:
                total_weight += FG.get_edge_data(q2, node)['weight']
        x['q1q2_inter' + suffix] = total_weight

        return x

    data = data.progress_apply(get_weights_adj, axis=1, raw=True)
    return data[['q1q2_inter' + suffix,'q1q2_union'+suffix]]

def get_shortest_path_worker(data,FG,suffix):

    def get_shortest_path(x):
        q1 = x['q1']
        q2 = x['q2']
        w = FG.get_edge_data(q1, q2)['weight']
        FG.remove_edge(q1, q2)
        try:
            res = nx.dijkstra_path_length(FG, q1, q2)
        except:
            res = 0
        FG.add_edge(q1, q2, weight=w)
        x['shortest_path'+suffix] = res
        return x
    data = data.progress_apply(get_shortest_path, axis=1, raw=True)
    return data['shortest_path'+suffix]

def graph_feature(train,test,use_label):

    def q_weight(data,FG,suffix):
        all_q_weights = {k: sum([x[1].get('weight') for x in FG[k].items()]) for k in FG.nodes}
        data['q1_num_adj' + suffix] = data['q1'].map(all_q_weights)
        data['q2_num_adj' + suffix] = data['q2'].map(all_q_weights)
        return data

    def multi_process(data,FG,suffix,feat_f):

        num_cpu = mlp.cpu_count()
        pool = mlp.Pool(num_cpu)

        aver_t = int(len(data) / num_cpu) + 1
        results = []
        for i in range(num_cpu):
            result = pool.apply_async(feat_f,args=(data.iloc[i*aver_t:(i+1)*aver_t],FG,suffix))
            results.append(result)
        pool.close()
        pool.join()

        feat = []
        for result in results:
            feat.append(result.get())
        feat = pd.concat(feat,axis=0)
        data = pd.concat([data,feat],axis=1)

        return data

    def pagerank(data,FG,suffix):
        pr = nx.pagerank(FG, alpha=0.85)
        data['q1_pr' + suffix] = data['q1'].map(pr)
        data['q2_pr' + suffix] = data['q2'].map(pr)
        return data

    if use_label:
        suffix = '_w'
    else:
        suffix = ''

    if use_label:
        train['y_pre'] = pd.read_csv('./data/tr_graph_weight.csv')['y_pre']
        test['y_pre'] = pd.read_csv('./data/te_graph_weight.csv')['y_pre']
    else:
        train['y_pre'] = 1.0
        test['y_pre'] = 1.0

    data = pd.concat([train, test],ignore_index=True)

    FG = nx.Graph()
    FG.add_weighted_edges_from(data[['q1','q2','y_pre']].values)

    data = pagerank(data,FG,suffix)
    data = q_weight(data,FG,suffix)

    data = multi_process(data, FG, suffix, get_shortest_path_worker)
    data = multi_process(data,FG,suffix,adj_feat_worker)

    data.drop(['y_pre'],inplace=True,axis=1)
    train = data.iloc[:train.shape[0]].reset_index( drop=True)
    test = data.iloc[train.shape[0]:].reset_index( drop=True)

    return train,test


def svd_graph(train,test,use_label):
    from scipy.sparse import coo_matrix,save_npz

    # train = pd.read_csv('./data/train.csv')
    # test = pd.read_csv('./data/test.csv')

    if use_label:
        train['y_pre'] = pd.read_csv('./data/tr_graph_weight.csv')['y_pre']
        test['y_pre'] = pd.read_csv('./data/te_graph_weight.csv')['y_pre']
    else:
        train['y_pre'] = 1.0
        test['y_pre'] = 1.0

    all_samples = pd.concat([train,test]).reset_index(drop=True)[['q1','q2','y_pre']]
    questions = all_samples['q1'].append(all_samples['q2']).drop_duplicates().reset_index(drop=True)


    q2i = pd.Series(questions.index.values, index=questions.values).to_dict()
    i2q = questions.to_dict()

    print('get coo matrix')
    row = [i for i in range(len(q2i))]
    col = [i for i in range(len(q2i))]
    value = [1 for i in range(len(q2i))]
    # row = []
    # col = []
    # value = []
    for q1,q2,w in all_samples.values:
        row.append(q2i[q1])
        col.append(q2i[q2])
        value.append(w)

        row.append(q2i[q2])
        col.append(q2i[q1])
        value.append(w)

    qmatrix = coo_matrix((value, (row,col)), shape=(len(q2i),len(q2i)))
    # save_npz('./data/q_adj_matrix.npz', qmatrix)


    print('svd ...')
    from config import n_components
    svd = TruncatedSVD(n_components=n_components,algorithm='arpack',n_iter=100)
    q_matrix = svd.fit_transform(qmatrix)

    total_ratio = []
    ratio = 0
    for i in svd.explained_variance_ratio_:
        ratio += i
        total_ratio.append(ratio)
    print(total_ratio)

    q_matrix[q_matrix<1e-5] = 0
    print(np.sum(q_matrix==0)/(q_matrix.shape[0]*q_matrix.shape[1]))
    q_matrix = pd.DataFrame(q_matrix,columns=['feat'+str(i) for i in range(n_components)])
    q_matrix['qid'] = list(range(len(q2i)))
    q_matrix['qid'] = q_matrix['qid'].map(i2q)

    q_matrix.to_csv('./data/q_matrix.csv',index=False)

    train.drop(['y_pre'],inplace=True,axis=1)
    test.drop(['y_pre'], inplace=True, axis=1)
    return q_matrix

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
labels = train['label']
# train.drop(['label'],axis=1,inplace=True)

# q_matrix = svd_graph(train,test,False)

# train,test = hash_q(train,test)
# train,test = graph_feature(train,test,True)
# train,test = graph_feature(train,test,False)
# train['label'] = labels




# train.to_csv('./data/train.csv',index=False)
# test.to_csv('./data/test.csv',index=False)

# train = pd.merge(train,q_matrix, left_on=['q1'], right_on=['qid'], how='left')
# test = pd.merge(test,q_matrix, left_on=['q1'], right_on=['qid'], how='left')

test['label'] = pd.read_csv('147037.csv')['y_pre']


print(train.corr())
print(test.corr())