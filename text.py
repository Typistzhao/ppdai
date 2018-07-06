import os
from config import use_device
#os.environ["CUDA_VISIBLE_DEVICES"] = use_device
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)
from keras import backend as K
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping,ModelCheckpoint,Callback,LearningRateScheduler
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import log_loss

def lr_de(epoch,lr):
    if epoch==0:
        return lr
    elif lr>0.0002:
            return lr/2
    else:
        return lr

class epochHistory(Callback):

    def on_train_begin(self, logs=None):
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)

def iter_ense(epochs,model,te):

    result = 0
    for e in epochs[-3:]:
        model.load_weights('./weight/weights.'+str(e+1)+'.hdf5')
        result += model.predict(te, batch_size=1024)
    return result/3

def train(use_data,semi_sv,output,data_aug,epoch=1000):
    def get_subset(dataset,idx):
        data = {}
        for key,value in dataset.items():
            data[key] = value[idx]
        return data

    def concat_data(data1,data2):
        result = {}
        for k in data1.keys():
            result[k] = np.concatenate([data1[k],data2[k]])
        return result

    from readdata import read_data


    tr,te, embedding_matrix, labels = read_data(use_data,data_aug=data_aug)

    print(use_data)
    print('Shape of label tensor:', labels.shape)

    y = labels

    from config import model_path
    from sklearn.cross_validation import StratifiedKFold, KFold
    from config import n_folds

    y_pred = pd.read_csv("./data/y_pred.csv")['y_pre'].values
    y_pos_ = y_pred == 1
    y_neg_ = y_pred == 0
    add_idx = np.any([y_pos_, y_neg_], axis=0)
    add_y = y_pred[add_idx]


    y_pos = y_pred > 0.75
    y_neg = y_pred < 0.25
    y_idx = np.any([y_pos, y_neg], axis=0)
    y_pred = y_pred[y_idx]
    print(y_idx.shape)


    folds = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
    result = np.zeros((len(te['q1']), 1))

    oof_y = np.zeros((len(y), 1))
    for n_fold, (tr_idx, val_idx) in enumerate(folds):
        tr_x = get_subset(tr,tr_idx)


        if semi_sv:
            te_x = get_subset(te, y_idx)
            tr_data = concat_data(tr_x,te_x)
            tr_y = np.concatenate([y[tr_idx],y_pred])
        else:
            add_data = get_subset(te,add_idx)
            tr_data = concat_data(tr_x,add_data)
            tr_y = np.concatenate([y[tr_idx], add_y])
            # tr_data = tr_x
            # tr_y = y[tr_idx]

        val_x = get_subset(tr, val_idx)
        val_y = y[val_idx]

        use_word = True
        if use_data!='words':
            use_word = False
        model = get_model(word_embedding_matrix=embedding_matrix,use_word=use_word)
        if n_fold == 0:
            print(model.summary())

        hist = epochHistory()
        print(n_fold)
        model.fit(tr_data,
                  tr_y,
                  epochs=epoch,
                  validation_data=[val_x,val_y],
                  verbose=1,
                  batch_size=256,
                  callbacks=[
                      EarlyStopping(patience=2, monitor='val_binary_crossentropy'),
                      # LearningRateScheduler(lr_de,verbose=1)
                      hist,
                      ModelCheckpoint('./weight/weights.{epoch:d}.hdf5',monitor='val_binary_crossentropy',save_weights_only=True)
                  ])
        result += iter_ense(hist.epochs,model,te)
        # result += model.predict(te, batch_size=1024)

        oof_y[val_idx] = model.predict(val_x, batch_size=2048)

        K.clear_session()
        tf.reset_default_graph()

    # 提交结果
    result /= n_folds
    submit = pd.DataFrame()
    submit['y_pre'] = list(result[:, 0])
    submit.to_csv(output, index=False)


    ## 保存预测的训练标签
    # oof_y = oof_y[:,0]
    # oof_y_ = oof_y.round().astype(int)
    #
    # error_idx = oof_y_!=y
    # print(np.sum(error_idx))
    # oof_y[error_idx] = 1-oof_y[error_idx]
    submit = pd.DataFrame()
    submit['y_pre'] = oof_y[:,0]
    submit.to_csv('./data/oofy.csv',index=False)

def train_wc(semi_sv,output,epoch=1000):
    from readdata import read_data

    tr_q1, tr_q2, te_q1, te_q2, word_embedding_matrix, labels = read_data('words')
    trc_q1, trc_q2, tec_q1, tec_q2, char_embedding_matrix, labels = read_data('chars')

    X = {
        'q1': tr_q1,
        'q2': tr_q2,
        'qc1': trc_q1,
        'qc2': trc_q2
    }
    y = labels


    from config import model_path
    from sklearn.cross_validation import StratifiedKFold, KFold
    from config import n_folds
    from nn import aggmodel

    y_pred = pd.read_csv("./data/y_pred.csv")['y_pre'].values
    y_pos = y_pred > 0.75
    y_neg = y_pred < 0.25
    y_idx = np.any([y_pos, y_neg], axis=0)
    y_pred = y_pred[y_idx]
    print(y_idx.shape)

    # oof_y = np.zeros((len(X['q1']),1))
    oof_y = pd.read_csv("./data/oofy.csv")['y_pre'].values
    alpha = 1
    oof_y = (1 - alpha) * y + alpha * oof_y


    folds = StratifiedKFold(y, n_folds=n_folds, shuffle=True,)
    result = np.zeros((len(te_q1), 1))

    for n_fold, (tr_idx, val_idx) in enumerate(folds):
        if semi_sv:
            Q1_tr = np.concatenate([X['q1'][tr_idx], te_q1[y_idx]])
            Q2_tr = np.concatenate([X['q2'][tr_idx], te_q2[y_idx]])
            Qc1_tr = np.concatenate([X['qc1'][tr_idx],tec_q1[y_idx]])
            Qc2_tr = np.concatenate([X['qc2'][tr_idx],tec_q2[y_idx]])
            y_tr = np.concatenate([y[tr_idx], y_pred])
            # y_tr = np.concatenate([oof_y[tr_idx], y_pred])

            idx = list(range(len(y_tr)))
            np.random.shuffle(idx)
            Q1_tr = Q1_tr[idx]
            Q2_tr = Q2_tr[idx]
            Qc1_tr = Qc1_tr[idx]
            Qc2_tr = Qc2_tr[idx]
            y_tr = y_tr[idx]
        else:
            Q1_tr = X['q1'][tr_idx]
            Q2_tr = X['q2'][tr_idx]
            Qc1_tr = X['qc1'][tr_idx]
            Qc2_tr = X['qc2'][tr_idx]
            y_tr = y[tr_idx]
            # y_tr = oof_y[tr_idx]



        Q1_te = X['q1'][val_idx]
        Q2_te = X['q2'][val_idx]
        Qc1_te = X['qc1'][val_idx]
        Qc2_te = X['qc2'][val_idx]
        y_te = y[val_idx]

        model = aggmodel(word_embedding_matrix,char_embedding_matrix)
        if n_fold == 0:
            print(model.summary())
        print(n_fold)
        model.fit([Q1_tr, Q2_tr,Qc1_tr,Qc2_tr],
                  y_tr,
                  epochs=epoch,
                  validation_data=[[Q1_te, Q2_te,Qc1_te, Qc2_te], y_te],
                  verbose=1,
                  batch_size=256,
                  callbacks=[
                      EarlyStopping(patience=3, monitor='val_binary_crossentropy'),
                      # LearningRateScheduler(lr_de,verbose=1)
                  ], )
        # model.load_weights(model_path)

        result += model.predict([te_q1,te_q2,tec_q1,tec_q2], batch_size=1024)


        # 释放显存
        K.clear_session()
        tf.reset_default_graph()

    # 提交结果
    result /= n_folds
    submit = pd.DataFrame()
    submit['y_pre'] = list(result[:, 0])
    submit.to_csv(output, index=False)


    ## 保存预测的训练标签
    # oof_y = oof_y[:,0]
    # oof_y_ = oof_y.round().astype(int)
    #
    # error_idx = oof_y_!=y
    # print(np.sum(error_idx))
    # oof_y[error_idx] = 1-oof_y[error_idx]
    # submit = pd.DataFrame()
    # submit['y_pre'] = oof_y
    # submit.to_csv('./data/oofy.csv',index=False)


from nn import rnnword,cnnword,matchPyramid,aggmodel
from config import use_model
if use_model=='rnnword':
    get_model = rnnword
elif use_model=='cnnword':
    get_model = cnnword
elif use_model=='matchPyramid':
    get_model = matchPyramid
elif use_model == 'aggmodel':
    pass
else:
    raise RuntimeError("don't have this model")

# for i in range(8):
#     print(" no semi   ",i,'-----------------------------------------------------','\n\n')
#     train(X, y,False, '0_'+str(i)+'.csv')

# train_wc(False,'agg.csv')
# train('word',False,'word0.csv',1000)
# train('chars',False,'char0_0.csv',1000)
# train('chars',False,'char0_1.csv',1000)


# train('words',False,'word0_0.csv',1000)
# train('words',False,'word0_1.csv',1000)

train('words',False,'word0_0.csv',False,1000)
train('words',True,'word1_0.csv',False,1000)
train('chars',False,'char0_0.csv',False,1000)
train('chars',True,'char1_0.csv',False,1000)




# train('words',True,'word1_1.csv',False,1000)
# train('words',False,'word0_1.csv',False,1000)
# train('chars',True,'char1_1.csv',False,1000)
# train('chars',False,'char0_1.csv',False,1000)

# train('chars',True,'char1_1.csv',1000)
# train('chars',True,'char1_2.csv',1000)
# train('chars',True,'char1_3.csv',1000)
# #
# train('words',True,'word1_0.csv',1000)
# # train('words',True,'word1_1.csv',1000)
# train('words',True,'word1_2.csv',1000)

# train_wc(False,'wc0.csv')
# train_wc(True,'wc1.csv')
# train(X,y,True,'base2.csv',1000)
# train(X,y,True,'base3.csv',1000)



