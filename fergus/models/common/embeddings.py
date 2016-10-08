from ikelos.layers import DynamicEmbedding, LambdaMask
import keras.backend as K
from keras.layers import Convolution2D, Lambda, Flatten, DataLayer, Embedding
from keras.regularizers import l2
from keras.engine import merge, Layer
import os
concat = lambda layers: merge(layers, mode='concat')

def make_convolutional_embedding(igor):
    p_emb = igor.p_emb_dropout
    p_W = igor.p_W_dropout
    p_U = igor.p_U_dropout
    w_decay = igor.weight_decay
    p_mlp = igor.p_mlp_dropout
    p_type = p_emb * igor.total_num_type / igor.total_num_pos # weight by the ratio to even things out
    Z_spinepos_layer = DataLayer(igor.X_spinepos, input_dtype='int32', name='spine_pos_lexicon')
    Z_spinetype_layer = DataLayer(igor.X_spinetype, input_dtype='int32', name='spine_type_lexicon')

    Z_spinepos = Z_spinepos_layer.tensor
    Z_spinetype = Z_spinetype_layer.tensor
    igor.preloaded_data = [Z_spinepos, Z_spinetype]

    F_embedpos = Embedding(igor.total_num_pos, igor.pos_embedding_size, mask_zero=True,
                           W_regularizer=l2(w_decay), dropout=p_emb, name='embed_pos')
    F_embedtype = Embedding(igor.total_num_type, igor.type_embedding_size, mask_zero=True,
                           W_regularizer=l2(w_decay), dropout=p_type, name='embed_type')

    F_stripmask = LambdaMask(lambda x, mask: None, name='strip_mask')
    spine_pos = F_stripmask(F_embedpos(Z_spinepos))
    spine_type = F_stripmask(F_embedtype(Z_spinetype))

    L = igor.spine_lexicon_size
    Sp = igor.max_spine_length
    C = igor.max_context_size
    poty = igor.pos_embedding_size + igor.type_embedding_size

    spine_cat = concat([spine_pos, spine_type])
    spine_cat = Lambda(lambda xin: K.permute_dimensions(xin, (0,3,1,2)), output_shape=lambda *args:(L, poty, Sp, C))(spine_cat)
    spine_conv1 = Convolution2D(igor.spine_convsame_filters, 1, 2, border_mode='valid')(spine_cat) 
    spine_conv2 = Convolution2D(igor.spine_convsame_filters, 2, 1, border_mode='valid')(spine_conv1)
    spine_conv3 = Convolution2D(igor.spine_convsame_filters, 1, 3, border_mode='valid')(spine_conv2)
    spine_conv4 = Convolution2D(igor.spine_convsame_filters, 3, 1, border_mode='valid')(spine_conv3)
    spine_conv5 = Convolution2D(igor.spine_convsame_filters, 4, 5, border_mode='valid')(spine_conv4)

    condensed_spine = Flatten()(spine_conv5)

    igor.logger.info("+ Making convolutional embeddings")
    igor.F_embedspine = DynamicEmbedding(condensed_spine,
                                         W_regularizer=l2(w_decay), # Output: Spine lexicon x Feature
                                         dropout=p_emb,
                                         name='embed_spine') 


def make_shallow_convolutional_embedding(igor):
    p_emb = igor.p_emb_dropout
    p_W = igor.p_W_dropout
    p_U = igor.p_U_dropout
    w_decay = igor.weight_decay
    p_mlp = igor.p_mlp_dropout
    p_type = p_emb * igor.total_num_type / igor.total_num_pos # weight by the ratio to even things out
    Z_spinepos_layer = DataLayer(igor.X_spinepos, input_dtype='int32', name='spine_pos_lexicon')
    Z_spinetype_layer = DataLayer(igor.X_spinetype, input_dtype='int32', name='spine_type_lexicon')

    Z_spinepos = Z_spinepos_layer.tensor
    Z_spinetype = Z_spinetype_layer.tensor
    igor.preloaded_data = [Z_spinepos, Z_spinetype]

    F_embedpos = Embedding(igor.total_num_pos, igor.pos_embedding_size, mask_zero=True,
                           W_regularizer=l2(w_decay), dropout=p_emb, name='embed_pos')
    F_embedtype = Embedding(igor.total_num_type, igor.type_embedding_size, mask_zero=True,
                           W_regularizer=l2(w_decay), dropout=p_type, name='embed_type')

    F_stripmask = LambdaMask(lambda x, mask: None, name='strip_mask')
    spine_pos = F_stripmask(F_embedpos(Z_spinepos))
    spine_type = F_stripmask(F_embedtype(Z_spinetype))

    L = igor.spine_lexicon_size
    Sp = igor.max_spine_length
    C = igor.max_context_size
    poty = igor.pos_embedding_size + igor.type_embedding_size

    spine_cat = concat([spine_pos, spine_type])
    spine_cat = Lambda(lambda xin: K.permute_dimensions(xin, (0,3,1,2)), output_shape=lambda *args:(L, poty, Sp, C))(spine_cat)
    spine_conv1 = Convolution2D(igor.spine_convsame_filters, 1, C, border_mode='valid')(spine_cat) 
    spine_conv2 = Convolution2D(igor.spine_convsame_filters, Sp, 1, border_mode='valid')(spine_conv1)

    condensed_spine = Flatten()(spine_conv2)

    igor.logger.info("+ Making shallow convolutional embeddings")
    igor.F_embedspine = DynamicEmbedding(condensed_spine,
                                         W_regularizer=l2(w_decay), # Output: Spine lexicon x Feature
                                         dropout=p_emb,
                                         name='embed_spine') 

def make_token_embedding(igor):
    igor.logger.info("+ Making token embeddings")
    igor.F_embedspine = Embedding(igor.spine_lexicon_size, 
                                  igor.spine_convsame_filters, 
                                  mask_zero=True, 
                                  W_regularizer=l2(igor.weight_decay), 
                                  name='token_embedding')
    igor.preloaded_data = []

def make_token_onehots(igor):
    igor.logger.info("+ Making token one-hots")
    igor.F_embedspine = OneHOtEmbedding(igor.spine_lexicon_size, name='tokenonehot_embeddng')
    igor.preloaded_data = []

class OneHotEmbedding(Layer):
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        super(OneHotEmbedding, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = K.eye(self.input_dim)
        
    def compute_mask(self, x, mask=None):
        return K.not_equal(x, 0)
    
    def get_output_shape_for(self, input_shape):
        return tuple(input_shape[:-1])+(self.input_dim,)
    
    def call(self, x, mask=None):
        out = K.gather(self.W, x)
        out *= K.expand_dims(K.cast(self.compute_mask(x, mask), K.floatx())) # for that numerical stability
        return out