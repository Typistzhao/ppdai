from keras.models import Model
from keras.layers import *
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import Nadam,RMSprop
import tensorflow as tf
from keras.initializers import VarianceScaling
from itertools import combinations


def co_attention(input1, input2, dim):

    transform = TimeDistributed(Dense(dim,use_bias=False))
    q1 = transform(input1)
    q2 = transform(input2)

    atten = Lambda(lambda x: K.batch_dot(x[0], x[1]))([q1, Permute((2, 1))(q2)])   # 15 * 15

    atten_2 = TimeDistributed(Activation('softmax'))(atten)
    atten_1 = TimeDistributed(Activation('softmax'))(Permute((2,1))(atten))

    q1 = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_2,q2])   # 15 * 15
    q2 = Lambda(lambda x: K.batch_dot(x[0], x[1]))([atten_1,q1])   # 15 * 15


    return q1, q2

def norm_layer(x, axis=1):
    return (x - K.mean(x, axis=axis, keepdims=True)) / K.std(x, axis=axis, keepdims=True)

def distance(q1,q2,dist,normlize=False):
    if normlize:
        q1 = Lambda(norm_layer)(q1)
        q2 = Lambda(norm_layer)(q2)

    if dist == 'cos':
        return multiply([q1,q2])

    elif dist == 'h_mean':
        def dice(x):
            return x[0]*x[1]/(K.sum(K.abs(x[0]),axis=1,keepdims=True)+K.sum(K.abs(x[1]),axis=1,keepdims=True))
        return Lambda(dice)([q1,q2])

    elif dist == 'dice':
        def dice(x):
            return x[0]*x[1]/(K.sum(x[0]**2,axis=1,keepdims=True)+K.sum(x[1]**2,axis=1,keepdims=True))
        return Lambda(dice)([q1,q2])

    elif dist == 'jaccard':
        def jaccard(x):
            return  x[0]*x[1]/(
                    K.sum(x[0]**2,axis=1,keepdims=True)+
                    K.sum(x[1]**2,axis=1,keepdims=True)-
                    K.sum(K.abs(x[0]*x[1]),axis=1,keepdims=True))
        return Lambda(jaccard)([q1,q2])

def pool_corr(q1,q2,pool_way):
    if pool_way == 'max':
        pool = GlobalMaxPooling1D()
    elif pool_way == 'ave':
        pool = GlobalAveragePooling1D()
    else:
        raise RuntimeError("don't have this pool way")

    q1 = pool(q1)
    q2 = pool(q2)

    merged = distance(q1,q2,'jaccard',normlize=True)


    return merged

def weight_ave(q1,q2):

    down = TimeDistributed(Dense(1,use_bias=False))

    q1 = down(Permute((2,1))(q1))
    q1 = Flatten()(q1)
    q1 = Lambda(norm_layer)(q1)
    q2 = down(Permute((2,1))(q2))
    q2 = Flatten()(q2)
    q2 = Lambda(norm_layer)(q2)
    merged = multiply([q1, q2])
    return merged

def simility_vec(q1,q2):
    simi = Lambda(lambda x: K.batch_dot(x[0], x[1]))([q1, Permute((2, 1))(q2)])
    simi = Reshape((-1,))(simi)
    return simi

def rnnword(word_embedding_matrix,use_word):
    if use_word:
        from config import MAX_NUM_WORDS
        text_len = MAX_NUM_WORDS
    else:
        from config import MAX_NUM_CHARS
        text_len = MAX_NUM_CHARS

    question1 = Input(shape=(text_len,),name='q1')
    question2 = Input(shape=(text_len,),name='q2')
    # q1_g = Input(shape=(text_len,))
    # q2_g = Input()


    embedd_word = Embedding(
                   len(word_embedding_matrix),
                   word_embedding_matrix.shape[1],
                   weights=[word_embedding_matrix],
                   input_length=text_len,
                   trainable=True)


    gru_dim1 = 384
    gru_dim2 = 256


    gru_w = Bidirectional(CuDNNGRU(gru_dim1,return_sequences=True),merge_mode='sum')
    gru2_w = Bidirectional(CuDNNGRU(gru_dim2,return_sequences=True,),merge_mode='sum')


    norm = BatchNormalization()
    q1 = embedd_word(question1)
    q1 = norm(q1)
    q1 = SpatialDropout1D(0.2)(q1)

    q2 = embedd_word(question2)
    q2 = norm(q2)
    q2 = SpatialDropout1D(0.2)(q2)

    q1_1 = gru_w(q1)
    q2_1 = gru_w(q2)

    q1 = gru2_w(q1_1)
    q2 = gru2_w(q2_1)

    merged_max = pool_corr(q1,q2,'max')
    merged_ave = pool_corr(q1,q2,'ave')
    # simi_vec = simility_vec(q1,q2)
    from config import n_components
    q1_g = Input(shape=(n_components,),name='q1node')
    q2_g = Input(shape=(n_components,), name='q2node')


    norm = BatchNormalization()
    q1_node = norm(q1_g)
    q2_node = norm(q2_g)

    fc = Dense(units=2)
    act = PReLU()
    q1_node = fc(q1_node)
    q1_node = act(q1_node)
    q2_node = fc(q2_node)
    q2_node = act(q2_node)

    node_vec = multiply([q1_node,q2_node])

    graph_f = Input(shape=(11,),name='gf')
    gf = BatchNormalization()(graph_f)
    gf = Dropout(0.3)(gf)
    # gf = Dense(18,activation='relu')(gf)

    merged = concatenate([merged_ave,merged_max])
    merged = Dense(512,activation='relu')(merged)
    merged = concatenate([merged, gf,node_vec])
    merged = Dense(512,activation='relu')(merged)
    output = Dense(1, activation='sigmoid')(merged)

    lr=0.0008

    model = Model(inputs=[question1,question2,graph_f,q1_g,q2_g], outputs=output)

    model.compile(loss='binary_crossentropy',optimizer=Nadam(lr),metrics=['binary_crossentropy','accuracy'])
    # model.load_weights("./data/weights_best_0.0008.hdf5")
    print(lr)

    return model


def cnnword(word_embedding_matrix, use_word):
    if use_word:
        from config import MAX_NUM_WORDS
        text_len = MAX_NUM_WORDS
    else:
        from config import MAX_NUM_CHARS
        text_len = MAX_NUM_CHARS

    def block_wrap(q1, q2, filters, k):
        conv = Conv1D(filters, kernel_size=k, padding='same')
        bn = BatchNormalization()
        act = Activation('relu')
        q1 = conv(q1)
        q1 = bn(q1)
        q1 = act(q1)
        q2 = conv(q2)
        q2 = bn(q2)
        q2 = act(q2)
        return q1, q2

    question1 = Input(shape=(text_len,), name='q1')
    question2 = Input(shape=(text_len,), name='q2')

    embedd_word = Embedding(len(word_embedding_matrix),
                            word_embedding_matrix.shape[1],
                            weights=[word_embedding_matrix],
                            input_length=text_len,
                            trainable=True,
                            # embeddings_regularizer=l2(0.000001)
                            )

    cnn_dim1 = 384
    cnn_dim2 = 256

    q1 = embedd_word(question1)
    q2 = embedd_word(question2)

    norm = BatchNormalization()
    q1 = norm(q1)
    q1 = SpatialDropout1D(0.2)(q1)

    q2 = norm(q2)
    q2 = SpatialDropout1D(0.2)(q2)


    q1_cnn_4, q2_cnn_4 = block_wrap(q1, q2, cnn_dim1,4)
    q1_cnn_4,q2_cnn_4 = MaxPool1D(pool_size=3,strides=2,padding='same')(q1_cnn_4),\
                        MaxPool1D(pool_size=3,strides=2,padding='same')(q2_cnn_4)
    q1_cnn_4, q2_cnn_4 = block_wrap(q1_cnn_4, q2_cnn_4, cnn_dim2,3)

    pool_layers_1 = pool_corr(q1_cnn_4, q2_cnn_4,'max')
    pool_layers_2 = pool_corr(q1_cnn_4, q2_cnn_4, 'ave')

    merged = concatenate([pool_layers_1,pool_layers_2])

    graph_f = Input(shape=(8,), name='gf')
    gf = BatchNormalization()(graph_f)
    gf = Dropout(0.2)(gf)

    merged = Dense(512, activation='relu')(merged)
    merged = concatenate([merged,gf])
    merged = Dense(512, activation='relu')(merged)

    output = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2,graph_f], outputs=output)

    model.compile(loss='binary_crossentropy', optimizer=Nadam(0.0008), metrics=['binary_crossentropy', 'accuracy'])

    return model


def matchPyramid(word_embedding_matrix,use_word):

    def cross(q1,q2,layer=None):
        if layer is not None:
            q1 = layer(q1)
            q2 = layer(q2)
        cr = Dot(axes=[2,2],normalize=True)([q1,q2])
        cr = Reshape((15, 15, 1))(cr)
        return cr

    def agg(merged):

        merged_1 = merged
        filters = 64
        k = 5

        conv1 = Conv2D(filters,k, strides=1,activation='relu')
        pool = MaxPool2D(pool_size=3, strides=2)
        # conv2 = Conv2D(filters,3, strides=1, activation='relu')

        merged_1 = conv1(merged_1)
        merged_1 = pool(merged_1)
        merged_1 = Flatten()(merged_1)

        return merged_1

    if use_word:
        from config import MAX_NUM_WORDS
        text_len = MAX_NUM_WORDS
    else:
        from config import MAX_NUM_CHARS
        text_len = MAX_NUM_CHARS

    question1 = Input(shape=(text_len,))
    question2 = Input(shape=(text_len,))

    embedd_word = Embedding(
        len(word_embedding_matrix),
        word_embedding_matrix.shape[1],
        weights=[word_embedding_matrix],
        input_length=text_len,
        trainable=True,
    )
    norm = BatchNormalization()
    q1 = embedd_word(question1)
    q1 = norm(q1)
    q1 = SpatialDropout1D(0.2)(q1)

    q2 = embedd_word(question2)
    q2 = norm(q2)
    q2 = SpatialDropout1D(0.2)(q2)


    channel = [cross(q1,q2)]

    dim = 512

    conv5 = Conv1D(dim, kernel_size=5, padding='same')
    conv7 = Conv1D(dim, kernel_size=7, padding='same')
    gru = Bidirectional(CuDNNGRU(dim, return_sequences=True), merge_mode='sum')
    lstm = Bidirectional(CuDNNLSTM(dim, return_sequences=True), merge_mode='sum')

    for layer in [conv5,conv7,gru,lstm]:
        channel.append(cross(q1, q2,layer))


    merged = concatenate(channel)

    merged = agg(merged)
    output = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1,question2], outputs=output)

    lr = 0.0008

    model.compile(loss='binary_crossentropy', optimizer=Nadam(lr), metrics=['binary_crossentropy', 'accuracy'])
    print(lr)
    return model

def aggmodel(word_embedding_matrix,char_embedding_matrix):

    def prepocess(q1,q2,embedd):
        norm = BatchNormalization()
        q1 = embedd(q1)
        q1 = norm(q1)
        q1 = SpatialDropout1D(0.2)(q1)

        q2 = embedd(q2)
        q2 = norm(q2)
        q2 = SpatialDropout1D(0.2)(q2)
        return q1,q2

    from config import MAX_NUM_WORDS,MAX_NUM_CHARS


    word1 = Input(shape=(MAX_NUM_WORDS,))
    word2 = Input(shape=(MAX_NUM_WORDS,))
    char1 = Input(shape=(MAX_NUM_CHARS,))
    char2 = Input(shape=(MAX_NUM_CHARS,))


    embedd_word = Embedding(
                   len(word_embedding_matrix),
                   word_embedding_matrix.shape[1],
                   weights=[word_embedding_matrix],
                   input_length=MAX_NUM_WORDS,
                   trainable=True)
    embedd_char = Embedding(
        len(char_embedding_matrix),
        char_embedding_matrix.shape[1],
        weights=[char_embedding_matrix],
        input_length=MAX_NUM_CHARS,
        trainable=True)

    gru_dim1 = 384
    gru_dim2 = 256


    gru_w = Bidirectional(CuDNNGRU(gru_dim1,return_sequences=True),merge_mode='sum')
    gru2_w = Bidirectional(CuDNNGRU(gru_dim2,return_sequences=True,),merge_mode='sum')

    gru_wc = Bidirectional(CuDNNGRU(gru_dim1, return_sequences=True), merge_mode='sum')
    gru2_wc = Bidirectional(CuDNNGRU(gru_dim2, return_sequences=True, ), merge_mode='sum')

    q1,q2 = prepocess(word1,word2,embedd_word)
    qc1,qc2 = prepocess(char1,char2,embedd_char)

    q1 = gru_w(q1)
    q2 = gru_w(q2)
    qc1 = gru_wc(qc1)
    qc2 = gru_wc(qc2)

    q1 = gru2_w(q1)
    q2 = gru2_w(q2)
    qc1 = gru2_wc(qc1)
    qc2 = gru2_wc(qc2)

    merged_max1 = pool_corr(q1,qc2,'max')
    merged_max2 = pool_corr(qc1,q2,'max')
    merged_ave1 = pool_corr(q1,qc2,'ave')
    merged_ave2 = pool_corr(qc1,q2,'ave')

    merged_max3 = pool_corr(q1,q2, 'max')
    merged_max4 = pool_corr(qc1,qc2, 'max')
    merged_ave3 = pool_corr(q1,q2, 'ave')
    merged_ave4 = pool_corr(qc1,qc2, 'ave')


    merged = concatenate([merged_max1,merged_max2,merged_max3,merged_max4,
                          merged_ave1,merged_ave2,merged_ave3,merged_ave4])
    merged = Dense(512,activation='relu')(merged)
    # merged = Dropout(0.2)(merged)
    merged = Dense(512,activation='relu')(merged)
    # merged = Dropout(0.2)(merged)
    output = Dense(1, activation='sigmoid')(merged)



    lr=0.0008


    model = Model(inputs=[word1,word2,char1,char2], outputs=output)

    # model = multi_gpu_model(model,gpus=4)

    model.compile(loss='binary_crossentropy',optimizer=Nadam(lr),metrics=['binary_crossentropy','accuracy'])

    # model.load_weights("./data/weights_best_0.0008.hdf5")
    print(lr)

    return model








