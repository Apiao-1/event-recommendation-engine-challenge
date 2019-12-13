# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""

from collections import OrderedDict, namedtuple
from itertools import chain

from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import  Embedding, Input, Flatten
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.utils import plot_model

import numpy as np


# 定长特征
# dimension = input type指输入的种类数，输入的字典长度
# embedding表示是否使用嵌入
# use_hash表示是否使用hash编码，设置这个值为true则不需要额外的编码操作，否则需要用LabelEncoder编码
class SparseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype','embedding_name','embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False, dtype="int32", embedding_name=None,embedding=True):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name,embedding)

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        if self.name == other.name and self.embedding_name == other.embedding_name:
            return True
        return False

    def __repr__(self):
        return 'SparseFeat:'+self.name

class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):

        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False

    def __repr__(self):
        return 'DenseFeat:'+self.name

# 不定长特征（如输入的字符串) https://deepctr-doc.readthedocs.io/en/latest/Examples.html#multi-value-input-movielens
class VarLenSparseFeat(namedtuple('VarLenFeat', ['name', 'dimension', 'maxlen', 'combiner', 'use_hash', 'dtype','weight_name','embedding_name','embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, maxlen,combiner="mean", use_hash=False, dtype="float32", weight_name=None,embedding_name=None,embedding=True):
        if embedding_name is None:
            embedding_name = name
        return super(VarLenSparseFeat, cls).__new__(cls, name, dimension, maxlen, combiner, use_hash, dtype,weight_name, embedding_name,embedding)

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False

    def __repr__(self):
        return 'VarLenSparseFeat:'+self.name


def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())

def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))

def build_input_features(feature_columns, mask_zero=True, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc,SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix+fc.name, dtype=fc.dtype)
        elif isinstance(fc,DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc,VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if not mask_zero:
                input_features[fc.name + "_seq_length"] = Input(shape=(
                    1,), name=prefix + 'seq_length_' + fc.name)
                input_features[fc.name + "_seq_max_length"] = fc.maxlen
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen,1),name=prefix + fc.weight_name ,dtype="float32")

        else:
            raise TypeError("Invalid feature column type,got",type(fc))

    return input_features


def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    if embedding_size == 'auto':
        print("Notice:Do not use auto embedding in models other than DCN")
        sparse_embedding = {feat.embedding_name: Embedding(feat.dimension, 6 * int(pow(feat.dimension, 0.25)),
                                                 embeddings_initializer=RandomNormal(
                                                     mean=0.0, stddev=init_std, seed=seed),
                                                 embeddings_regularizer=l2(l2_reg),
                                                 name=prefix + '_emb_' + feat.name) for feat in
                            sparse_feature_columns}
    else:

        sparse_embedding = {feat.embedding_name: Embedding(feat.dimension, embedding_size,
                                                 embeddings_initializer=RandomNormal(
                                                     mean=0.0, stddev=init_std, seed=seed),
                                                 embeddings_regularizer=l2(
                                                     l2_reg),
                                                 name=prefix + '_emb_'  + feat.name) for feat in
                            sparse_feature_columns}

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            if embedding_size == "auto":
                sparse_embedding[feat.embedding_name] = Embedding(feat.dimension, 6 * int(pow(feat.dimension, 0.25)),
                                                        embeddings_initializer=RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=l2(
                                                            l2_reg),
                                                        name=prefix + '_seq_emb_' + feat.name,
                                                        mask_zero=seq_mask_zero)

            else:
                sparse_embedding[feat.embedding_name] = Embedding(feat.dimension, embedding_size,
                                                        embeddings_initializer=RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=l2(
                                                            l2_reg),
                                                        name=prefix + '_seq_emb_' + feat.name,
                                                        mask_zero=seq_mask_zero)
    return sparse_embedding

def get_embedding_vec_list(embedding_dict, input_dict, sparse_feature_columns, return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fg in sparse_feature_columns:
        feat_name = fg.name
        if len(return_feat_list) == 0  or feat_name in return_feat_list:
            if fg.use_hash:
                lookup_idx = Hash(fg.dimension,mask_zero=(feat_name in mask_feat_list))(input_dict[feat_name])
            else:
                lookup_idx = input_dict[feat_name]

            embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))

    return embedding_vec_list


# 构建所有使用嵌入的sparse特征的embedding层，dict形式
def create_embedding_matrix(feature_columns,l2_reg,init_std,seed,embedding_size, prefix="",seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat) and x.embedding, feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat) and x.embedding, feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std, seed,
                                                 l2_reg, prefix=prefix + 'sparse',seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict

def get_linear_logit(features, feature_columns, units=1, use_bias=False, init_std=0.0001, seed=1024, prefix='linear',
                     l2_reg=0):

    linear_emb_list = [input_from_feature_columns(features,feature_columns,1,l2_reg,init_std,seed,prefix=prefix+str(i))[0] for i in range(units)]
    _, dense_input_list = input_from_feature_columns(features,feature_columns,1,l2_reg,init_std,seed,prefix=prefix)

    linear_logit_list = []
    for i in range(units):

        if len(linear_emb_list[0])>0 and len(dense_input_list) >0:
            sparse_input = concat_fun(linear_emb_list[i])
            dense_input = concat_fun(dense_input_list)
            linear_logit = Linear(l2_reg,mode=2,use_bias=use_bias)([sparse_input,dense_input])
        elif len(linear_emb_list[0])>0: # 只有sparse特征
            sparse_input = concat_fun(linear_emb_list[i])
            linear_logit = Linear(l2_reg,mode=0,use_bias=use_bias)(sparse_input)
        elif len(dense_input_list) >0: # 只有dense特征
            dense_input = concat_fun(dense_input_list)
            linear_logit = Linear(l2_reg,mode=1,use_bias=use_bias)(dense_input)
        else:
            raise NotImplementedError
        linear_logit_list.append(linear_logit)

    return concat_fun(linear_logit_list)

def embedding_lookup(sparse_embedding_dict,sparse_input_dict,sparse_feature_columns,return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0  or feature_name in return_feat_list ) and fc.embedding:
            if fc.use_hash:
                lookup_idx = Hash(fc.dimension,mask_zero=(feature_name in mask_feat_list))(sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            embedding_vec_list.append(sparse_embedding_dict[embedding_name](lookup_idx))

    return embedding_vec_list

# 从dict中找出存在的col，即使用embedding的col
def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.dimension, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict

def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns):
    pooling_vec_list = []
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = feature_name + '_seq_length'
        if feature_length_name in features:
            if fc.weight_name is not None:
                seq_input =WeightedSequenceLayer()([embedding_dict[feature_name],features[feature_length_name],features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
            [seq_input, features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input =WeightedSequenceLayer(supports_masking=True)([embedding_dict[feature_name],features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
            seq_input)
        pooling_vec_list.append(vec)
    return pooling_vec_list

def get_varlen_multiply_list(embedding_dict, features, varlen_sparse_feature_columns_name_dict):
    multiply_vec_list = []
    print(embedding_dict)
    for key_feature in varlen_sparse_feature_columns_name_dict:
        for value_feature in varlen_sparse_feature_columns_name_dict[key_feature]:
            key_feature_length_name = key_feature.name + '_seq_length'
            if isinstance(value_feature, VarLenSparseFeat):
                value_input = embedding_dict[value_feature.name]
            elif isinstance(value_feature, DenseFeat):
                value_input = features[value_feature.name]
            else:
                raise TypeError("Invalid feature column type,got",type(value_feature))
            if key_feature_length_name in features:
                varlen_vec = WeightedSequenceLayer()(
                    [embedding_dict[key_feature.name], features[key_feature_length_name], value_input])
                vec = SequencePoolingLayer('sum', supports_masking=False)(
                    [varlen_vec, features[key_feature_length_name]])
            else:
                varlen_vec = WeightedSequenceLayer(supports_masking=True)(
                    [embedding_dict[key_feature.name], value_input])
                vec = SequencePoolingLayer('sum', supports_masking=True)( varlen_vec)
            multiply_vec_list.append(vec)
    return multiply_vec_list

def get_dense_input(features,feature_columns):
    dense_feature_columns = list(filter(lambda x:isinstance(x,DenseFeat),feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list


def input_from_feature_columns(features,feature_columns, embedding_size, l2_reg, init_std, seed,prefix='',seq_mask_zero=True,support_dense=True):


    sparse_feature_columns = list(filter(lambda x:isinstance(x,SparseFeat),feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    # 构建所有使用嵌入的sparse特征的embedding层，dict形式
    embedding_dict = create_embedding_matrix(feature_columns,l2_reg,init_std,seed,embedding_size, prefix=prefix,seq_mask_zero=seq_mask_zero)
    sparse_embedding_list = embedding_lookup(
        embedding_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features,feature_columns)
    if not support_dense and len(dense_value_list) >0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    # 对所有不定长的进行池化
    sequence_embed_dict = varlen_embedding_lookup(embedding_dict,features,varlen_sparse_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, varlen_sparse_feature_columns)
    sparse_embedding_list += sequence_embed_list

    return sparse_embedding_list, dense_value_list



def combined_dnn_input(sparse_embedding_list,dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_fun(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_fun(dense_value_list))
        return concat_fun([sparse_dnn_input,dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_fun(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_fun(dense_value_list))
    else:
        raise NotImplementedError

class Hash(tf.keras.layers.Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        if x.dtype != tf.string:
            x = tf.as_string(x, )
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                                    name=None)  # weak hash
        except:
            hash_x = tf.strings.to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                               name=None)  # weak hash
        if self.mask_zero:
            mask_1 = tf.cast(tf.not_equal(x, "0"), 'int64')
            mask_2 = tf.cast(tf.not_equal(x, "0.0"), 'int64')
            mask = mask_1 * mask_2
            hash_x = (hash_x + 1) * mask
        return hash_x

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero}
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Linear(tf.keras.layers.Layer):

    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, **kwargs):

        self.l2_reg = l2_reg
        # self.l2_reg = tf.contrib.layers.l2_regularizer(float(l2_reg_linear))
        if mode not in [0,1,2]:
            raise ValueError("mode must be 0,1 or 2")
        self.mode = mode
        self.use_bias = use_bias
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        if self.mode != 0 :
            self.dense = tf.keras.layers.Dense(units=1, activation=None, use_bias=False,
                                           kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))

        super(Linear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs , **kwargs):

        if self.mode == 0:
            sparse_input = inputs
            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            linear_logit = self.dense(dense_input)
        else:
            sparse_input, dense_input = inputs

            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keep_dims=False) + self.dense(dense_input)
        if self.use_bias:
            linear_logit += self.bias

        return linear_logit

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'l2_reg': self.l2_reg}
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def concat_fun(inputs, axis=-1,mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)

class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None

from tensorflow.python.keras.layers import LSTM, Lambda, Layer

class WeightedSequenceLayer(Layer):
    """The WeightedSequenceLayer is used to apply weight score on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len,seq_weight]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

        - seq_weight is a 3D tensor with shape: ``(batch_size, T, 1)``

      Output shape
        - 3D tensor with shape: ``(batch_size, T, embedding_size)``.

      Arguments
        - **weight_normalization**: bool.Whether normalize the weight socre before applying to sequence.

        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self,weight_normalization=False, supports_masking=False, **kwargs):
        super(WeightedSequenceLayer, self).__init__(**kwargs)
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(WeightedSequenceLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, input_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            key_input, value_input = input_list
            mask = tf.expand_dims(mask[0], axis=2)
        else:
            key_input, key_length_input, value_input = input_list
            mask = tf.sequence_mask(key_length_input,
                                    self.seq_len_max, dtype=tf.bool)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = key_input.shape[-1]

        if self.weight_normalization:
            paddings = tf.ones_like(value_input) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(value_input)
        value_input = tf.where(mask, value_input, paddings)

        if self.weight_normalization:
           value_input = tf.nn.softmax(value_input,dim=1)


        if len(value_input.shape) == 2:
            value_input = tf.expand_dims(value_input, axis=2)
            value_input = tf.tile(value_input, [1, 1, embedding_size])

        return tf.multiply(key_input,value_input)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask):
        if self.supports_masking:
            return mask[0]
        else:
            return None

    def get_config(self, ):
        config = {'supports_masking': self.supports_masking}
        base_config = super(WeightedSequenceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SequencePoolingLayer(Layer):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.

        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8,tf.float32)
        super(SequencePoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask,tf.float32)#                tf.to_float(mask)
            user_behavior_length = tf.reduce_sum(mask, axis=-1, keep_dims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list

            mask = tf.sequence_mask(user_behavior_length,
                                    self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]

        mask = tf.tile(mask, [1, 1, embedding_size])

        uiseq_embed_list *= mask
        hist = uiseq_embed_list
        if self.mode == "max":
            return tf.reduce_max(hist, 1, keep_dims=True)

        hist = tf.reduce_sum(hist, 1, keep_dims=False)

        if self.mode == "mean":
            hist = tf.div(hist, tf.cast(user_behavior_length,tf.float32) + self.eps)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return (None, 1, input_shape[-1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def gen_sequence(dim, max_len, sample_size):
    return np.array([np.random.randint(0, dim, max_len) for _ in range(sample_size)]), np.random.randint(1, max_len + 1, sample_size)

def get_test_data(sample_size=1000, sparse_feature_num=1, dense_feature_num=1, sequence_feature=['sum', 'mean', 'max'],
                  classification=True, include_length=False, hash_flag=False,prefix=''):


    feature_columns = []
    model_input = {}


    if 'weight'  in sequence_feature:
        feature_columns.append(VarLenSparseFeat(prefix+"weighted_seq",2,3,weight_name=prefix+"weight"))
        feature_columns.append(
                    SparseFeat(prefix+"weighted_seq_seq_length", 1,embedding=False))

        s_input, s_len_input = gen_sequence(
            2, 3, sample_size)

        model_input[prefix+"weighted_seq"] = s_input
        model_input[prefix+'weight'] = np.random.randn(sample_size,3,1)
        model_input[prefix+"weighted_seq"+"_seq_length"] = s_len_input
        sequence_feature.pop(sequence_feature.index('weight'))


    for i in range(sparse_feature_num):
        dim = np.random.randint(1, 10)
        feature_columns.append(SparseFeat(prefix+'sparse_feature_'+str(i), dim,hash_flag,tf.int32))
    for i in range(dense_feature_num):
        feature_columns.append(DenseFeat(prefix+'dense_feature_'+str(i), 1,tf.float32))
    for i, mode in enumerate(sequence_feature):
        dim = np.random.randint(1, 10)
        maxlen = np.random.randint(1, 10)
        feature_columns.append(
            VarLenSparseFeat(prefix+'sequence_' + str(i), dim, maxlen, mode))



    for fc in feature_columns:
        if isinstance(fc,SparseFeat):
            model_input[fc.name]=np.random.randint(0, fc.dimension, sample_size)
        elif isinstance(fc,DenseFeat):
            model_input[fc.name] = np.random.random(sample_size)
        else:
            s_input, s_len_input = gen_sequence(
                fc.dimension, fc.maxlen, sample_size)
            model_input[fc.name] = s_input
            if include_length:
                feature_columns.append(
                    SparseFeat(prefix+'sequence_' + str(i)+'_seq_length', 1,embedding=False))
                model_input[prefix+"sequence_"+str(i)+'_seq_length'] = s_len_input






    if classification:
        y = np.random.randint(0, 2, sample_size)
    else:
        y = np.random.random(sample_size)

    return model_input, y, feature_columns

def check_model(model, model_name, x, y,check_model_io=True):
    """
    compile model,train and evaluate it,then save/load weight and model file.
    :param model:
    :param model_name:
    :param x:
    :param y:
    :param check_model_io: test save/load model file or not
    :return:
    """

    # # 设定格式化模型名称，以时间戳作为标记
    # model_name = "test"
    # # 设定存储位置，每个模型不一样的路径
    # tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))


    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    # model.fit(x, y, batch_size=100, epochs=1, validation_split=0.5, callbacks=[tensorboard])
    model.fit(x, y, batch_size=100, epochs=1, validation_split=0.5)

    print(model_name+" test train valid pass!")

    print(model_name + " test pass!")

    # plot_model(model, to_file='1.png')
    plot_model(model, to_file=model_name+'.png')

import sys
def activation_layer(activation):
    if (isinstance(activation, str)) or (sys.version_info.major == 2 and isinstance(activation, (str, unicode))):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer


class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss

         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DNN(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate,seed=self.seed+i) for i in range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            # fc = Dense(self.hidden_size[i], activation=None, \
            #           kernel_initializer=glorot_normal(seed=self.seed), \
            #           kernel_regularizer=l2(self.l2_reg))(deep_input)
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc,training = training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def FNN(linear_feature_columns, dnn_feature_columns, embedding_size=8, dnn_hidden_units=(128, 128),
        l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0,
        dnn_activation='relu', task='binary'):
    """Instantiates the Factorization-supported Neural Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param embedding_size: positive integer,sparse feature embedding_size
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear weight
    :param l2_reg_dnn: float . L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """
    # 将所有特征转化成tensor，返回的是OrderedDict
    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    # 从OrderedDict中获取所有的value，即获取所有的tensor
    inputs_list = list(features.values())

    # 得到输入dnn的特征list，在fnn中不区分两者，因为dnn_feature_columns = linear_feature_columns
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features,dnn_feature_columns,
                                                                              embedding_size,
                                                                              l2_reg_embedding,init_std,
                                                                              seed)

    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    dnn_input = combined_dnn_input(sparse_embedding_list,dense_value_list)
    deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn,
                   dnn_dropout, False, seed)(dnn_input)
    deep_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(deep_out)
    final_logit = tf.keras.layers.add([deep_logit, linear_logit])
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=output)
    return model


def FNN_test(sparse_feature_num, dense_feature_num):

    model_name = "FNN"

    sample_size = 8
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num, dense_feature_num)

    model = FNN(feature_columns,feature_columns, dnn_hidden_units=[32, 32], dnn_dropout=0.5)
    check_model(model, model_name, x, y)

if __name__ == '__main__':
    FNN_test(1,1)