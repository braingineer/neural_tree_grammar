from ikelos.layers import DynamicEmbedding, LambdaMask
import keras.backend as K
from keras.layers import Convolution2D, Lambda, Flatten, DataLayer, Embedding
from keras.regularizers import l2
from keras.engine import merge
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

    igor.logger.info(" + Making convolutional embeddings")
    igor.F_embedspine = DynamicEmbedding(condensed_spine,
                                         W_regularizer=l2(w_decay), # Output: Spine lexicon x Feature
                                         dropout=p_emb,
                                         name='embed_spine') 


def make_token_embedding(igor):
        igor.logger.info(" + Making token embeddings")
        igor.F_embedspine = Embedding(igor.spine_lexicon_size, 
                                      igor.spine_convsame_filters, 
                                      mask_zero=True, 
                                      W_regularizer=l2(igor.weight_decay), 
                                      name='token_embedding')
        igor.preloaded_data = []
    

def make_logger(igor, loggername):
    import logging
    igor.logfile_level = 'debug'
    igor.logshell_level = 'info'
    igor.logfile_location = '.'
    logger = logging.getLogger(loggername)
    levels = {"debug": logging.DEBUG, "warning":logging.WARNING,
              "info": logging.INFO, "error":logging.ERROR,
              "critical":logging.CRITICAL}
    logger.setLevel(logging.DEBUG)

    if igor.logfile_level != "off":
        safe_loc = igor.logfile_location+("/" if igor.logfile_location[-1] != "/" else "")
        if not os.path.exists(safe_loc):
            os.makedirs(safe_loc)
        fh = logging.FileHandler(os.path.join(igor.logfile_location, 
                                              "{}.debug.log".format(loggername)))
        fh.setLevel(levels[igor.logfile_level])
        logger.addHandler(fh)

    if igor.logshell_level != "off":
        sh = logging.StreamHandler()
        sh.setLevel(levels[igor.logfile_level])
        logger.addHandler(sh)

    return logger