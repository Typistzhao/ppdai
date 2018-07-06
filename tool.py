import pandas as pd
from tqdm import tqdm
import json
import numpy as np

def cluster_pos(file='train'):


    tr = pd.read_csv('./data/'+file+'.csv')
    # tr = tr.append(pd.read_csv('save_sample.csv')).reset_index(drop=True)
    y_idx = tr['label'] == 1


    # y_pred = pd.read_csv('./149367.csv')['y_pre'].values
    # y_pos = y_pred < 1
    # y_neg = y_pred > 0.5
    # y_idx = np.logical_and(y_pos,y_neg)


    tr = tr.loc[y_idx,['q1','q2']]
    print(tr.shape)
    tr = tr.sort_values(by=['q1', 'q2']).values
    for i in range(len(tr)):
        if tr[i][0]>tr[i][1]:
            tr[i] = [tr[i][1],tr[i][0]]

    q2group = {}
    num_group = 0
    error = 0
    for q1,q2 in tr:
        assert q1 < q2
        if q1 not in q2group and q2 not in q2group:
            q2group[q1] = num_group
            q2group[q2] = num_group
            num_group += 1
        elif q2 in q2group and q1 not in q2group:
            q2group[q1] = q2group[q2]
        elif q1 in q2group and q2 not in q2group:
            q2group[q2] = q2group[q1]
        else:
            if q2group[q2] != q2group[q1]:
                error+=1
    print(error)
    while error != 0:
        for q1,q2 in tr:
            if q2group[q1] != q2group[q2]:
                group_id = min(q2group[q1],q2group[q2])
                q2group[q1] = group_id
                q2group[q2] = group_id
        error = 0
        for q1,q2 in tr:
            if q2group[q1] != q2group[q2]:
                error+=1
        print(error)


    with open('./info/q2group.json','w') as f:
        f.write(json.dumps(q2group,sort_keys=True,indent=4, separators=(',', ': ')))

    group2q = [{} for i in range(num_group)]
    for q,g_id in q2group.items():
        group2q[g_id][q] = 1

    with open('./info/group2q.json', 'w') as f:
        f.write(json.dumps(group2q, sort_keys=True, indent=4, separators=(',', ': ')))


    group_n = {}
    for i,q in enumerate(group2q):
        group_n[str(i)] = len(q)
    group_n = sorted(group_n.items(),key=lambda x:x[1])
    with open('./info/group_samples_num.json', 'w') as f:
        f.write(json.dumps(group_n, sort_keys=True, indent=4, separators=(',', ': ')))



def cluster_neg():

    tr = pd.read_csv('./data/train.csv')
    tr = tr.loc[tr['label'] == 0, ['q1', 'q2']].values

    with open('./info/q2group.json','r') as f:
        q2group = json.loads(f.read())

    neg_pair = {}
    for q1,q2 in tr:
        if q1 in q2group and q2 in q2group:
            if q2group[q1]<q2group[q2]:
                neg_pair[str(q2group[q1])+'_'+str(q2group[q2])] = 1
            elif q2group[q1]>q2group[q2]:
                neg_pair[str(q2group[q2])+'_'+str(q2group[q1])] = 1

    with open('./info/neg_rule.json','w') as f:
        f.write(json.dumps(neg_pair,sort_keys=True,indent=4,separators=(',', ': ')))

    te = pd.read_csv('./data/test.csv').values
    need_rule = {}
    for q1, q2 in te:
        if q1 in q2group and q2 in q2group:
            if q2group[q1] < q2group[q2]:
                pair = str(q2group[q1]) + '_' + str(q2group[q2])
            elif q2group[q1] > q2group[q2]:
                pair = str(q2group[q2]) + '_' + str(q2group[q1])
            else:
                continue
            if pair not in neg_pair:
                if pair not in need_rule:
                    need_rule[pair] = 0
                need_rule[pair]+=1
    need_rule = sorted(need_rule.items(),key=lambda x:x[1])
    with open('./info/need_rule.json','w') as f:
        f.write(json.dumps(need_rule,sort_keys=True,indent=4,separators=(',', ': ')))



def create_pos_sample():

    with open('./info/q_te_dict.json','r') as f:
        q_te = json.loads(f.read())
    with open('./info/q_tr_dict.json','r') as f:
        q_tr = json.loads(f.read())

    with open('./info/group2q.json','r') as f:
        group2q = json.loads(f.read())

    from itertools import combinations

    samples_dict = {}

    for questions in tqdm(group2q):
        if len(questions) == 2:
            continue
        if len(questions) == 0:
            continue
        for q1,q2 in combinations(list(questions.keys()),2):
            samples_dict[q1 + '_' + q2] = 1


    tr = pd.read_csv('./data/train.csv')
    te = pd.read_csv('./data/test.csv')
    tr.append(te)
    tr = tr[['q1','q2']].values
    for q1,q2 in tr:
        a = q1 + '_' + q2 in samples_dict
        b = q2 + '_' + q1 in samples_dict
        assert (a and b) == False
        if q1 + '_' + q2 in samples_dict:
            samples_dict.pop(q1 + '_' + q2)
        elif q2 + '_' + q1 in samples_dict:
            samples_dict.pop(q2 + '_' + q1)

    samples = []
    for k in samples_dict.keys():
        samples.append(k.split("_"))

    print(len(samples))

    train_extend = pd.DataFrame(samples,columns=['q1','q2'])
    train_extend.to_csv('./info/pos_sample.csv',index=False)


def create_neg_sample():

    with open('./info/group2q.json','r') as f:
        group2q = json.loads(f.read())
    with open('./info/neg_rule.json','r') as f:
        neg_pair = json.loads(f.read())

    from itertools import product

    samples_dict = {}
    num_sample = 0
    for pair in tqdm(neg_pair):
        c1,c2 = pair.split('_')
        for q1,q2 in product(group2q[int(c1)],group2q[int(c2)]):
            samples_dict[q1 + '_' + q2] = 1
            num_sample+=1

    print(num_sample)
    tr = pd.read_csv('./data/train.csv',usecols=['q1','q2'])
    te = pd.read_csv('./data/test.csv',usecols=['q1','q2'])
    tr.append(te)
    tr = tr[['q1','q2']].values
    for q1,q2 in tr:
        a = q1 + '_' + q2 in samples_dict
        b = q2 + '_' + q1 in samples_dict
        assert (a and b) == False
        if q1 + '_' + q2 in samples_dict:
            samples_dict.pop(q1 + '_' + q2)
        elif q2 + '_' + q1 in samples_dict:
            samples_dict.pop(q2 + '_' + q1)

    samples = []
    for k in samples_dict.keys():
        samples.append(k.split("_"))

    print(len(samples))
    del samples_dict
    import gc
    gc.collect()
    train_extend = pd.DataFrame(samples,columns=['q1','q2'])
    train_extend.to_csv('./info/neg_sample.csv',index=False)


def post_process(file,output='baseline.csv'):

    with open('./info/q2group.json','r') as f:
        q2group = json.loads(f.read())

    with open('./info/group2q.json','r') as f:
        group2q = json.loads(f.read())

    te = pd.read_csv('./data/test.csv',usecols=['q1','q2']).values
    y_pre = pd.read_csv(file)


    "正例修正"
    n = 0
    loss = 0

    save_samples = []
    s = 0
    for i, (q1, q2) in enumerate(te):
        if q1 in q2group and q2 in q2group:
            if q2group[q1] == q2group[q2]:
                n += 1
                loss = loss - np.log(y_pre.iloc[i,0])
                y_pre.iloc[i, 0] = 1
                save_samples.append([1,q1, q2])

    # save_samples = pd.DataFrame(save_samples,columns=['label','q1','q2'])
    # save_samples.to_csv('./info/save_sample.csv',index=False)
    print('n:',n)

    print(s)
    "负例修正"
    with open('./info/neg_rule.json','r') as f:
        neg_pair = json.loads(f.read())
    n = 0
    for i, (q1, q2) in tqdm(enumerate(te)):
        if q1 in q2group and q2 in q2group:
            if q2group[q1] < q2group[q2]:
                pair = str(q2group[q1]) + '_' + str(q2group[q2])
            elif q2group[q1] > q2group[q2]:
                pair = str(q2group[q2]) + '_' + str(q2group[q1])
            else:
                pair = ''
            if pair in neg_pair:
                loss = loss - np.log(1-y_pre.iloc[i, 0])
                y_pre.iloc[i, 0] = 0
                n += 1
    print('loss:', loss / len(te))
    print(n)

    y_pre.to_csv(output, index=False)

def q_distr():

    te = pd.read_csv('./data/test.csv').values

    q_dict = {}
    for q1, q2 in te:
        if q1 not in q_dict:
            q_dict[q1] = 0
        q_dict[q1] += 1
        if q2 not in q_dict:
            q_dict[q2] = 0
        q_dict[q2] += 1
    te_q = sorted(q_dict.items(), key=lambda x: x[1])
    with open('./info/te_q.json', 'w') as f:
        f.write(json.dumps(te_q, sort_keys=True, indent=4, separators=(',', ': ')))
    with open('./info/q_te_dict.json', 'w') as f:
        f.write(json.dumps(q_dict, sort_keys=True, indent=4, separators=(',', ': ')))

    tr = pd.read_csv('./data/train.csv',usecols=['q1','q2']).values

    q_dict = {}
    for q1, q2 in tr:
        if q1 not in q_dict:
            q_dict[q1] = 0
        q_dict[q1] += 1
        if q2 not in q_dict:
            q_dict[q2] = 0
        q_dict[q2] += 1
    tr_q = sorted(q_dict.items(), key=lambda x: x[1])
    with open('./info/tr_q.json', 'w') as f:
        f.write(json.dumps(tr_q, sort_keys=True, indent=4, separators=(',', ': ')))
    with open('./info/q_tr_dict.json', 'w') as f:
        f.write(json.dumps(q_dict, sort_keys=True, indent=4, separators=(',', ': ')))

def te_test():

    with open('q2group.json','r') as f:
        q2group = json.loads(f.read())
    with open('neg_rule.json','r') as f:
        neg_pair = json.loads(f.read())

    te = pd.read_csv('./data/test.csv').values

def get_samples():

    with open('./info/q_te_dict.json', 'r') as f:
        q_te = json.loads(f.read())
    with open('./info/q_tr_dict.json', 'r') as f:
        q_tr = json.loads(f.read())

    pos_samples = pd.read_csv('./info/pos_sample.csv',usecols=['q1','q2']).sample(frac=1).reset_index(drop=True).values
    neg_samples = pd.read_csv('./info/neg_sample.csv',usecols=['q1','q2']).sample(frac=1).reset_index(drop=True).values

    data = []
    for samples in [pos_samples,neg_samples]:
        q_freq = {}
        num_sample = 0
        for q1,q2 in samples:
            if q1 not in q_te or q2 not in q_te:
                continue
            if q1 not in q_freq:
                q_freq[q1] = 0
            if q2 not in q_freq:
                q_freq[q2] = 0
            if q_freq[q1] > 4-q_tr[q1]/30+q_te[q1]/10:
                continue
            if q_freq[q2] > 4-q_tr[q2]/30+q_te[q2]/10:
                continue
            q_freq[q1] += 1
            q_freq[q2] += 1
            data.append([q1,q2])
            num_sample += 1
        print(num_sample)

    data = pd.DataFrame(data,columns=['q1','q2'])
    return data


if __name__=='__main__':
    # q_distr()

    create_pos_sample()
    create_neg_sample()
    # cluster_pos()
    # cluster_neg()
    # from glob import  glob
    # path= './new_result/'
    # files = glob(path+'*.csv')
    # for f in files:
    #     post_process(f,f)
    # post_process('./word1_1.csv')
    # post_process_v2()









