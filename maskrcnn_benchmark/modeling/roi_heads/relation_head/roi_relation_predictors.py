# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import copy
import logging
import numpy as np
import torch
from torch.nn.modules.activation import ReLU
from torch.nn.modules.container import Sequential
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.data import get_dataset_statistics


@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, mode="normal")
        layer_init(self.rel_compress, mode="xavier")
        layer_init(self.ctx_compress, mode="xavier")
        layer_init(self.post_cat, mode="xavier")
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, mode="xavier")
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
                
        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, mode="xavier")
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, mode="normal")
        layer_init(self.post_cat, mode="xavier")
        layer_init(self.rel_compress, mode="xavier")
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, mode="xavier")
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, mode="xavier")
        #layer_init(self.frq_gate, mode="xavier")
        layer_init(self.ctx_compress, mode="xavier")
        #layer_init(self.uni_compress, mode="xavier")

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, mode="normal")
        layer_init(self.post_cat, mode="xavier")
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, mode="xavier")
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True),])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, mode="xavier")
        
        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, mode="normal")
        if not self.use_vtranse:
            layer_init(self.post_cat[0], mode="xavier")
            layer_init(self.ctx_compress, mode="xavier")
        layer_init(self.vis_compress, mode="xavier")
        
        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])
            layer_init(self.spt_emb[0], mode="xavier")
            layer_init(self.spt_emb[2], mode="xavier")

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

        
    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append( head_rep[pair_idx[:,0]] - tail_rep[pair_idx[:,1]] )
            else:
                ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list
        
        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats
        
        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep  
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':   # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE': # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            #union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            #union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            #union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            #union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest
            
        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


@registry.ROI_RELATION_PREDICTOR.register("KnowledgeTransferPredictor")
class KnowledgeTransferPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(KnowledgeTransferPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.feat_mode = config.MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.FEAT_MODE 
        self.transfer = config.MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.KNOWLEDGE_TRANSFER
        self.vis_record = False
        self.feature_loss = config.MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.FEATURE_LOSS
        self.feat_path = config.MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.PRETRAINED_FEATURE_PATH
        # self.source = config.MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.KNOWLEDGE_SOURCE
        self.logger = logging.getLogger("maskrcnn_benchmark").getChild("predictor")

        if config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                mode = 'predcls'
            else:
                mode = 'sgcls'
        else:
            mode = 'sgdet'

        assert in_channels is not None

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics["obj_classes"], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init encoding
        if config.MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # pos decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.edge_dim = self.hidden_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                        nn.ReLU(inplace=True),])

        if self.feat_mode == "concat":
            assert self.use_bias
            self.feat_dim = self.pooling_dim+self.num_rel_cls
        elif self.feat_mode == "basic":
            self.feat_dim = self.pooling_dim
        else:
            raise ValueError("Invalid feature extractor. Please select from 'concat' or 'basic'")

        self.first_compress = nn.Linear(self.feat_dim, self.num_rel_cls)
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
        else:
            self.union_single_not_match = False

        if self.transfer:
            self.class_features = nn.Parameter(torch.zeros(self.num_rel_cls, self.feat_dim).to(torch.float), requires_grad=False)
            if self.feat_path:
                self.class_features = nn.Parameter(torch.from_numpy(np.load(self.feat_path, allow_pickle=True).item()["avg_feature"]).to(torch.float).cuda(), requires_grad=False)
            if self.feature_loss != "none":
                assert self.feature_loss=="mse" or self.feature_loss=="margin", "Please select 'mse' or 'margin' or 'none' for feature loss"
                self.class_features.requires_grad = True
                self.mseloss = nn.MSELoss()
                if self.feature_loss == "margin":
                    self.margin = 40.0
                    self.weight = 1 if mode == "sgdet" else 0.1

            self.final_compress = nn.Linear(self.feat_dim, self.num_rel_cls)

            # if self.source != ():
            #     pred2idx = {"and": 5, "says": 39, "belonging to": 9, "over": 33, "parked on": 35, "growing on": 18, "standing on": 41, "made of": 27, "attached to": 7, "at": 6, "in": 22, "hanging from": 19, "wears": 49, "in front of": 23, "from": 17, "for": 16, "watching": 47, "lying on": 26, "to": 42, "behind": 8, "flying in": 15, "looking at": 25, "on back of": 32, "holding": 21, "between": 10, "laying on": 24, "riding": 38, "has": 20, "across": 2, "wearing": 48, "walking on": 46, "eating": 14, "above": 1, "part of": 36, "walking in": 45, "sitting on": 40, "under": 43, "covered in": 12, "carrying": 11, "using": 44, "along": 4, "with": 50, "on": 31, "covering": 13, "of": 30, "against": 3, "playing": 37, "near": 29, "painted on": 34, "mounted on": 28}
            #     self.source_list = np.array(list(map(lambda x: pred2idx[x], list(self.source))), dtype=int)

            # hyperparameter for calibration
            self.alpha = config.MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.CALIBRATION_ALPHA


        self.layer_initialize()

        # for object class embedding 
        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)

    def layer_initialize(self):
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, mode="normal")
        layer_init(self.post_cat[0], mode="kaiming")
        if self.union_single_not_match:
            layer_init(self.up_dim, mode="xavier")
        layer_init(self.first_compress, mode="xavier")
        if self.transfer:
            layer_init(self.final_compress, mode="xavier")

    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        post_ctx_rep = self.post_cat(ctx_rep)


        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=False):
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        assert union_features != None

        # # object classes embedding
        if self.use_bias:
            freq_emb = self.freq_bias.index_with_labels(pair_pred.long())
        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        if self.feat_mode == "concat":
            # concat [object contextual features + union features, object classes embedding]
            feature = torch.cat([post_ctx_rep+union_features, freq_emb], dim=-1)
            rel_dist = self.first_compress(feature)
        elif self.feat_mode == "basic":
            feature = post_ctx_rep*union_features
            rel_dist = self.first_compress(feature)
            if self.use_bias and (not self.transfer):
                rel_dist = rel_dist + freq_emb
        
        final_dist_list = rel_dist.split(num_rels, dim=0)

        if self.vis_record:
            self.feature_record(feature, rel_labels)
            return None, None, {}

        if self.transfer:  
            p = F.softmax(rel_dist, -1)
            ###### you select specific knowledge source #######
            # if self.source != ():
            #     knowledge = torch.matmul(F.softmax(p[:,self.source_list], -1), self.class_features.detach()[self.source_list])
            # else:
            ###### knowledge calculation ######
            knowledge = torch.matmul(p, self.class_features.detach())
            ###### refine feature ######
            # attention selector --------------------------------
            attention = torch.clamp(torch.tanh(feature+knowledge), min=0)
            refined_feature = feature + attention*knowledge
            # ---------------------------------------------------

            # long tail feature calibration
            max_score, _ = p.max(dim=1)
            refined_feature = self.alpha*torch.mul(max_score.view(-1, 1), refined_feature)
            final_dist = self.final_compress(refined_feature)
            if self.feat_mode=="basic" and self.use_bias:
                final_dist = final_dist + freq_emb
            final_dist_list = final_dist.split(num_rels, dim=0)

        # TODO code for vstree binary loss
        add_losses = {}
        if self.training:
            if self.transfer:
                rel_labels = cat(rel_labels, dim=0)
                add_losses["first prediction"] = F.cross_entropy(rel_dist, rel_labels)

                feature_detached = feature.detach()
                if self.feature_loss == "mse":
                    add_losses["feature"] = self.mseloss(self.class_features[rel_labels], feature_detached)
                
                elif self.feature_loss == "margin":
                    batch_size = feature_detached.size(0)
                    # attract loss
                    counts = feature_detached.new_ones(self.num_rel_cls)
                    counts = counts.scatter_add_(0, rel_labels, counts.new_ones(batch_size))
                    class_feature_batch = self.class_features.index_select(0,rel_labels).to(feature_detached.dtype)
                    diff = (feature_detached - class_feature_batch).pow(2)/2
                    div = counts.index_select(0, rel_labels)
                    diff /= div.view(-1,1)
                    add_losses["feature attract loss"] = self.weight*diff.sum()/batch_size
                    # repel loss
                    dist_mat = torch.cdist(feature_detached, self.class_features.to(feature_detached.dtype))
                    classes = torch.arange(self.num_rel_cls).long().cuda()
                    labels_expand = rel_labels.unsqueeze(1).expand(batch_size, self.num_rel_cls)
                    mask = labels_expand.ne(classes.expand(batch_size, self.num_rel_cls)).int()
                    distmat_neg = torch.mul(dist_mat, mask)
                    # original attract:repel = 1:0.01
                    add_losses["feature repel loss"] = self.weight*0.01*torch.clamp(self.margin - distmat_neg.sum()/(batch_size*self.num_rel_cls), 0.0, 1e6)
            
                elif self.feature_loss=="none":
                    # moving average
                    with torch.no_grad():
                        for i in range(self.num_rel_cls):
                            if len(feature_detached[rel_labels==i]):
                                self.class_features[i] = 0.3*self.class_features[i]+0.7*(feature_detached[rel_labels==i]).mean(dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)
            
        return obj_dist_list, final_dist_list, add_losses
    
    def feature_record(self, feature, rel_labels):
        """
        Update feature mean, sum, squared sum
        """
        rel_labels = cat(rel_labels, dim=0).to('cpu').numpy()
        feature = feature.detach().to('cpu').numpy()
        for i in range(self.num_rel_cls):
            if len(feature[rel_labels==i]):
                self.feat["avg_feature"][i] = 0.3*self.feat["avg_feature"][i]+0.7*(feature[rel_labels==i]).mean(axis=0)

@registry.ROI_RELATION_PREDICTOR.register("PSKTPredictor")
class PSKTPredictor(nn.Module):
    def __init__(self, config, in_channels, taxonomy):
        super(PSKTPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.feat_mode = config.MODEL.ROI_RELATION_HEAD.PSKT.FEAT_MODE 
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.transfer = config.MODEL.ROI_RELATION_HEAD.PSKT.KNOWLEDGE_TRANSFER
        self.feat_path = config.MODEL.ROI_RELATION_HEAD.PSKT.PRETRAINED_FEATURE_PATH
        self.feature_loss = config.MODEL.ROI_RELATION_HEAD.PSKT.FEATURE_LOSS
        self.logger = logging.getLogger("maskrcnn_benchmark").getChild("predictor")

        if config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                mode = 'predcls'
            else:
                mode = 'sgcls'
        else:
            mode = 'sgdet'

        assert in_channels is not None

        # set taxonomy
        self.T = taxonomy
        self.num_tree = self.T["num_tree"]
        self.global2children = torch.from_numpy(self.T["global_to_children"]).cuda()
        self.is_ancestor_mat = torch.from_numpy(self.T["is_ancestor_mat"].astype(config.DTYPE)).cuda()
        self.num_children = self.T["num_children"]
        self.children_idxs = self.T["children"]
        self.children_tree_index = self.T["children_tree_index"]

        # # ignore "on"
        # self.is_ancestor_mat[:,31] = 0

        assert in_channels is not None 

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics["obj_classes"], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init encoding
        if config.MODEL.ROI_RELATION_HEAD.PSKT.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.PSKT.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # pos decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.edge_dim = self.hidden_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                        nn.ReLU(inplace=True),])
        if self.feat_mode == "concat":
            assert self.use_bias
            self.feat_dim = self.pooling_dim+self.num_rel_cls
        elif self.feat_mode == "basic":
            self.feat_dim = self.pooling_dim
        else:
            raise ValueError(f"{self.feat_mode} is invalid feature mode. Please select from 'concat' or 'basic'")

        self.first_compress = nn.ModuleList(nn.Linear(self.feat_dim, num) for num in self.num_children)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
        else:
            self.union_single_not_match = False

        if self.transfer:
            self.class_features = nn.Parameter(torch.zeros(self.num_rel_cls, self.feat_dim).to(torch.float), requires_grad=False)
            if self.feat_path:
                self.class_features = nn.Parameter(torch.from_numpy(np.load(self.feat_path, allow_pickle=True).item()["avg_feature"]).to(torch.float).cuda(), requires_grad=False)
            if self.feature_loss != "none":
                assert self.feature_loss=="mse" or self.feature_loss=="margin", "Please select 'mse' or 'margin' or 'none' for feature loss"
                self.class_features.requires_grad = True
                self.mseloss = nn.MSELoss()
                if self.feature_loss == "margin":
                    self.margin = 40.0
                    self.weight = 1 if mode == "sgdet" else 0.1

            self.final_compress = nn.ModuleList(nn.Linear(self.feat_dim, num) for num in self.num_children)
            #for calibration
            self.alpha = config.MODEL.ROI_RELATION_HEAD.PSKT.CALIBRATION_ALPHA
    

        # for object class embedding 
        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)
            if "basic" in self.feat_mode:
                self.freq_compress = nn.ModuleList(nn.Linear(self.num_rel_cls, num) for num in self.num_children)
        
        self.layer_initialize()


    def layer_initialize(self):
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, mode="normal")
        layer_init(self.post_cat[0], mode="kaiming")
        if self.union_single_not_match:
            layer_init(self.up_dim, mode="xavier")
        for i in range(self.num_tree):
            layer_init(self.first_compress[i], mode="xavier")
            if self.transfer:
                layer_init(self.final_compress[i], mode="xavier")
            if "basic" in self.feat_mode:
                layer_init(self.freq_compress[i], mode="xavier")
            

    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        post_ctx_rep = self.post_cat(ctx_rep)


        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list
    
    def knowledge_transfer(self, feature, tree_index):
        first_dist = self.first_compress[tree_index](feature)
        if not self.transfer:
            return first_dist, None
        # case: no kt in root -------------------
        if tree_index == 0:
            return first_dist, first_dist
        # ---------------------------------------
        p = F.softmax(first_dist, -1)
        max_score, max_idx = p.max(dim=1)
        #### knowledge calculation ####
        parent_feature = torch.matmul(self.is_ancestor_mat[self.children_idxs[tree_index]], self.class_features.detach())
        div_n = torch.sum(self.is_ancestor_mat[self.children_idxs[tree_index]], dim=1)
        parent_feature = parent_feature/div_n.view(-1,1)
        knowledge = torch.matmul(p, parent_feature)
        #### refine feature ####
        # attention selector -----------------------------------------
        attention = torch.clamp(torch.tanh(feature+knowledge), min=0)
        refined_feature = feature + attention*knowledge
        #-------------------------------------------------------------

        #### feature calibration ####
        refined_feature = self.alpha*torch.mul(max_score.view(-1,1), refined_feature)
        final_dist = self.final_compress[tree_index](refined_feature)
        return first_dist, final_dist


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=False):
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        assert union_features != None 

        # object classes embedding
        if self.use_bias:
            freq_emb = self.freq_bias.index_with_labels(pair_pred.long())

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)
        if self.feat_mode == "concat":
            feature = torch.cat([post_ctx_rep+union_features, freq_emb], dim=-1)
        elif self.feat_mode == "basic":
            feature = post_ctx_rep * union_features

        first_dists = []
        final_dists = []

        for i in range(self.num_tree):
            first_dist, final_dist = self.knowledge_transfer(feature, i)
            if (not self.transfer) and ("basic" in self.feat_mode) and self.use_bias and i!=0:
                # # freq info selector
                first_dist = first_dist + self.freq_compress[i](freq_emb)
                # freq info extraction
                # --- if you apply it to root ----
                # anc_mat = self.is_ancestor_mat[self.children_idxs[i]]
                # anc_mat /= torch.sum(anc_mat, dim=1).view(-1,1)
                # final_dist = final_dist + torch.matmul(freq_emb, torch.t(anc_mat))
                # --- else ----
                # first_dist = first_dist + torch.matmul(freq_emb, torch.t(self.is_ancestor_mat[self.children_idxs[i]]))
            first_dists.append(first_dist)
            if self.transfer:
                if ("basic" in self.feat_mode) and self.use_bias and i!=0:
                    # # freq info selector 
                    final_dist = final_dist + self.freq_compress[i](freq_emb)
                    # freq info extraction 
                    # --- if you apply it to root ----
                    # anc_mat = self.is_ancestor_mat[self.children_idxs[i]]
                    # anc_mat /= torch.sum(anc_mat, dim=1).view(-1,1)
                    # final_dist = final_dist + torch.matmul(freq_emb, torch.t(anc_mat))
                    # --- else ----
                    # final_dist = final_dist + torch.matmul(freq_emb, torch.t(self.is_ancestor_mat[self.children_idxs[i]])) #階層分類だと平均とる必要あり, クラスター選択タイプならいらない
                final_dists.append(final_dist)

        if self.transfer:
            final_dists_list = [(final_dists[i].split(num_rels, dim=0)) for i in range(self.num_tree)]
        else:
            final_dists_list = [(first_dists[i].split(num_rels, dim=0)) for i in range(self.num_tree)]

        # TODO code for vstree binary loss
        add_losses = {}
        if self.training:
            if self.transfer:
                rel_labels = cat(rel_labels, dim=0)
                feature_detached = feature.detach()
                if self.feature_loss == "mse":
                    add_losses["feature"] = self.mseloss(self.class_features[rel_labels], feature_detached)
                
                elif self.feature_loss == "margin":
                    batch_size = feature_detached.size(0)
                    # attract loss
                    counts = feature_detached.new_ones(self.num_rel_cls)
                    counts = counts.scatter_add_(0, rel_labels, counts.new_ones(batch_size))
                    class_feature_batch = self.class_features.index_select(0,rel_labels).to(feature_detached.dtype)
                    diff = (feature_detached - class_feature_batch).pow(2)/2
                    div = counts.index_select(0, rel_labels)
                    diff /= div.view(-1,1)
                    add_losses["feature attract loss"] = self.weight*diff.sum()/batch_size
                    # repel loss
                    dist_mat = torch.cdist(feature_detached, self.class_features.to(feature_detached.dtype))
                    classes = torch.arange(self.num_rel_cls).long().cuda()
                    labels_expand = rel_labels.unsqueeze(1).expand(batch_size, self.num_rel_cls)
                    mask = labels_expand.ne(classes.expand(batch_size, self.num_rel_cls)).int()
                    distmat_neg = torch.mul(dist_mat, mask)
                    # original attract:repel = 1:0.01
                    add_losses["feature repel loss"] = self.weight*0.01*torch.clamp(self.margin - distmat_neg.sum()/(batch_size*self.num_rel_cls), 0.0, 1e6)
                
                elif self.feature_loss=="none":
                    # moving average
                    with torch.no_grad():
                        for i in range(self.num_rel_cls):
                            if len(feature_detached[rel_labels==i]):
                                self.class_features[i] = 0.3*self.class_features[i]+0.7*(feature_detached[rel_labels==i]).mean(dim=0)

                # rel_labels for each cluster
                rel_label_chs = []
                for i in range(self.num_tree):
                    rel_label_ch = self.global2children[i][rel_labels]
                    rel_label_chs.append(rel_label_ch) 
                add_losses['first_prediction'] = torch.zeros([1]).cuda() # combined feature (ctx+union)
                div_n = 0
                for i in range(1, self.num_tree):
                    ####### CROSS ENTROPY LOSS FOR EACH TREE ########
                    if (rel_label_chs[i]>=0).sum():
                        div_n += 1
                        add_losses['first_prediction'] += F.cross_entropy(first_dists[i], rel_label_chs[i], ignore_index=-1) 
                # normalization
                if div_n:
                    add_losses["first_prediction"] /= div_n

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)
        
        return obj_dist_list, final_dists_list, add_losses

@registry.ROI_RELATION_PREDICTOR.register("PSKTAllPredictor")
class PSKTAllPredictor(nn.Module):
    def __init__(self, config, in_channels, taxonomy):
        super(PSKTAllPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.feat_mode = config.MODEL.ROI_RELATION_HEAD.PSKTALL.FEAT_MODE 
        self.transfer = config.MODEL.ROI_RELATION_HEAD.PSKTALL.KNOWLEDGE_TRANSFER
        self.feature_loss = config.MODEL.ROI_RELATION_HEAD.PSKTALL.FEATURE_LOSS
        self.feat_path = config.MODEL.ROI_RELATION_HEAD.PSKTALL.PRETRAINED_FEATURE_PATH
        self.logger = logging.getLogger("maskrcnn_benchmark").getChild("predictor")

        if config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                mode = 'predcls'
            else:
                mode = 'sgcls'
        else:
            mode = 'sgdet'

        assert in_channels is not None

        # set taxonomy
        self.T = taxonomy
        self.num_tree = self.T["num_tree"]
        self.global2children = torch.from_numpy(self.T["global_to_children"]).cuda()
        self.is_ancestor_mat = torch.from_numpy(self.T["is_ancestor_mat"].astype(config.DTYPE)).cuda()
        self.num_children = self.T["num_children"]
        self.children_idxs = self.T["children"]
        self.children_tree_index = self.T["children_tree_index"]

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics["obj_classes"], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init encoding
        if config.MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # pos decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.edge_dim = self.hidden_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                        nn.ReLU(inplace=True),])

        if self.feat_mode == "concat":
            assert self.use_bias
            self.feat_dim = self.pooling_dim+self.num_rel_cls
        elif self.feat_mode == "basic":
            self.feat_dim = self.pooling_dim
        else:
            raise ValueError("Invalid feature extractor. Please select from 'concat' or 'basic'")

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
        else:
            self.union_single_not_match = False

        self.final_compress = nn.ModuleList(nn.Linear(self.feat_dim, num) for num in self.num_children)

        if self.transfer:
            self.first_compress = nn.Linear(self.feat_dim, self.num_rel_cls)
            self.class_features = nn.Parameter(torch.zeros(self.num_rel_cls, self.feat_dim).to(torch.float), requires_grad=False)
            if self.feat_path:
                self.class_features = nn.Parameter(torch.from_numpy(np.load(self.feat_path, allow_pickle=True).item()["avg_feature"]).to(torch.float).cuda(), requires_grad=False)
            if self.feature_loss != "none":
                assert self.feature_loss=="mse" or self.feature_loss=="margin", "Please select 'mse' or 'margin' or 'none' for feature loss"
                self.class_features.requires_grad = True
                self.mseloss = nn.MSELoss()
                if self.feature_loss == "margin":
                    self.margin = 40.0
                    self.weight = 1 if mode == "sgdet" else 0.1

            # hyperparameter for calibration
            self.alpha = config.MODEL.ROI_RELATION_HEAD.PSKTALL.CALIBRATION_ALPHA

        self.layer_initialize()

        # for object class embedding 
        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)
            if "basic" in self.feat_mode:
                self.freq_compress = nn.ModuleList(nn.Linear(self.num_rel_cls, num) for num in self.num_children)

    def layer_initialize(self):
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, mode="normal")
        layer_init(self.post_cat[0], mode="kaiming")
        if self.union_single_not_match:
            layer_init(self.up_dim, mode="xavier")
        for i in range(self.num_tree):
            layer_init(self.final_compress[i], mode="xavier")
            if "basic" in self.feat_mode:
                layer_init(self.freq_compress[i])
        if self.transfer:
            layer_init(self.first_compress, mode="xavier")

    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        post_ctx_rep = self.post_cat(ctx_rep)


        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=False):
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        assert union_features != None

        # # object classes embedding
        if self.use_bias:
            freq_emb = self.freq_bias.index_with_labels(pair_pred.long())
        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        if self.feat_mode == "concat":
            # concat [object contextual features + union features, object classes embedding]
            feature = torch.cat([post_ctx_rep+union_features, freq_emb], dim=-1)
        elif self.feat_mode == "basic":
            feature = post_ctx_rep*union_features

        if self.transfer:
            first_dist = self.first_compress(feature)
 
            p = F.softmax(first_dist, -1)
            ###### knowledge calculation ######
            knowledge = torch.matmul(p, self.class_features.detach())
            ###### refine feature ######
            # attention selector --------------------------------
            attention = torch.clamp(torch.tanh(feature+knowledge), min=0)
            refined_feature = feature + attention*knowledge
            # ---------------------------------------------------

            # long tail feature calibration
            max_score, _ = p.max(dim=1)
            refined_feature = self.alpha*torch.mul(max_score.view(-1, 1), refined_feature)
        
        final_dists = []
        for i in range(self.num_tree):
            if self.transfer:
                final_dist = self.final_compress[i](refined_feature)
            else:
                final_dist = self.final_compress[i](feature)
            if ("basic" in self.feat_mode) and self.use_bias:
                # # freq info selector 
                final_dist = final_dist + self.freq_compress[i](freq_emb)
            final_dists.append(final_dist)

        final_dists_list = [(final_dists[i].split(num_rels, dim=0)) for i in range(self.num_tree)]

        # TODO code for vstree binary loss
        add_losses = {}
        if self.training:
            if self.transfer:
                rel_labels = cat(rel_labels, dim=0)
                add_losses["first prediction"] = F.cross_entropy(first_dist, rel_labels)

                feature_detached = feature.detach()
                if self.feature_loss == "mse":
                    add_losses["feature"] = self.mseloss(self.class_features[rel_labels], feature_detached)
                
                elif self.feature_loss == "margin":
                    batch_size = feature_detached.size(0)
                    # attract loss
                    counts = feature_detached.new_ones(self.num_rel_cls)
                    counts = counts.scatter_add_(0, rel_labels, counts.new_ones(batch_size))
                    class_feature_batch = self.class_features.index_select(0,rel_labels).to(feature_detached.dtype)
                    diff = (feature_detached - class_feature_batch).pow(2)/2
                    div = counts.index_select(0, rel_labels)
                    diff /= div.view(-1,1)
                    add_losses["feature attract loss"] = self.weight*diff.sum()/batch_size
                    # repel loss
                    dist_mat = torch.cdist(feature_detached, self.class_features.to(feature_detached.dtype))
                    classes = torch.arange(self.num_rel_cls).long().cuda()
                    labels_expand = rel_labels.unsqueeze(1).expand(batch_size, self.num_rel_cls)
                    mask = labels_expand.ne(classes.expand(batch_size, self.num_rel_cls)).int()
                    distmat_neg = torch.mul(dist_mat, mask)
                    # original attract:repel = 1:0.01
                    add_losses["feature repel loss"] = self.weight*0.01*torch.clamp(self.margin - distmat_neg.sum()/(batch_size*self.num_rel_cls), 0.0, 1e6)
            
                elif self.feature_loss=="none":
                    # moving average
                    with torch.no_grad():
                        for i in range(self.num_rel_cls):
                            if len(feature_detached[rel_labels==i]):
                                self.class_features[i] = 0.3*self.class_features[i]+0.7*(feature_detached[rel_labels==i]).mean(dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)
            
        return obj_dist_list, final_dists_list, add_losses

# rootだけ全ラベルKTして、他の分類器は類似ラベルKT
@registry.ROI_RELATION_PREDICTOR.register("PSKTRootAllPredictor")
class PSKTRootAllPredictor(nn.Module):
    def __init__(self, config, in_channels, taxonomy):
        super(PSKTRootAllPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.feat_mode = config.MODEL.ROI_RELATION_HEAD.PSKTROOTALL.FEAT_MODE 
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.feat_path = config.MODEL.ROI_RELATION_HEAD.PSKTROOTALL.PRETRAINED_FEATURE_PATH
        self.feature_loss = config.MODEL.ROI_RELATION_HEAD.PSKTROOTALL.FEATURE_LOSS
        self.logger = logging.getLogger("maskrcnn_benchmark").getChild("predictor")

        if config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                mode = 'predcls'
            else:
                mode = 'sgcls'
        else:
            mode = 'sgdet'

        assert in_channels is not None

        # set taxonomy
        self.T = taxonomy
        self.num_tree = self.T["num_tree"]
        self.global2children = torch.from_numpy(self.T["global_to_children"]).cuda()
        self.is_ancestor_mat = torch.from_numpy(self.T["is_ancestor_mat"].astype(config.DTYPE)).cuda()
        self.num_children = self.T["num_children"]
        self.children_idxs = self.T["children"]
        self.children_tree_index = self.T["children_tree_index"]

        # # ignore "on"
        # self.is_ancestor_mat[:,31] = 0

        assert in_channels is not None 

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics["obj_classes"], statistics['rel_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init encoding
        if config.MODEL.ROI_RELATION_HEAD.PSKTROOTALL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.PSKTROOTALL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # pos decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.edge_dim = self.hidden_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                        nn.ReLU(inplace=True),])
        if self.feat_mode == "concat":
            assert self.use_bias
            self.feat_dim = self.pooling_dim+self.num_rel_cls
        elif self.feat_mode == "basic":
            self.feat_dim = self.pooling_dim
        else:
            raise ValueError(f"{self.feat_mode} is invalid feature mode. Please select from 'concat' or 'basic'")

        self.first_compress = nn.ModuleList(nn.Linear(self.feat_dim, self.num_rel_cls) if i==0 else nn.Linear(self.feat_dim, num) for i, num in enumerate(self.num_children))

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
        else:
            self.union_single_not_match = False

        self.class_features = nn.Parameter(torch.zeros(self.num_rel_cls, self.feat_dim).to(torch.float), requires_grad=False)
        if self.feat_path:
            self.class_features = nn.Parameter(torch.from_numpy(np.load(self.feat_path, allow_pickle=True).item()["avg_feature"]).to(torch.float).cuda(), requires_grad=False)
        if self.feature_loss != "none":
            assert self.feature_loss=="mse" or self.feature_loss=="margin", "Please select 'mse' or 'margin' or 'none' for feature loss"
            self.class_features.requires_grad = True
            self.mseloss = nn.MSELoss()
            if self.feature_loss == "margin":
                self.margin = 40.0
                self.weight = 0.1 if mode == "sgdet" else 1

        self.final_compress = nn.ModuleList(nn.Linear(self.feat_dim, num) for num in self.num_children)
        #for calibration
        self.alpha = config.MODEL.ROI_RELATION_HEAD.PSKTROOTALL.CALIBRATION_ALPHA

        # feature encoder for root classification
        self.root_encoder = nn.Linear(self.feat_dim, self.feat_dim//2)
        self.final_compress[0] = nn.Linear(self.feat_dim//2, self.num_children[0])


        # for object class embedding 
        if self.use_bias:
            self.freq_bias = FrequencyBias(config, statistics)
            if "basic" in self.feat_mode:
                self.freq_compress = nn.ModuleList(nn.Linear(self.num_rel_cls, num) for num in self.num_children)
        
        self.layer_initialize()


    def layer_initialize(self):
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, mode="normal")
        layer_init(self.post_cat[0], mode="kaiming")
        layer_init(self.root_encoder, mode="xavier")
        if self.union_single_not_match:
            layer_init(self.up_dim, mode="xavier")
        for i in range(self.num_tree):
            layer_init(self.first_compress[i], mode="xavier")
            layer_init(self.final_compress[i], mode="xavier")
            if "basic" in self.feat_mode:
                layer_init(self.freq_compress[i], mode="xavier")
            

    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list
    
    
    def knowledge_transfer(self, feature, tree_index):
        first_dist = self.first_compress[tree_index](feature)
        p = F.softmax(first_dist, -1)
        max_score, max_idx = p.max(dim=1)
        #### knowledge calculation ####
        if tree_index == 0:
            parent_feature = self.class_features.detach()
        else:
            parent_feature = torch.matmul(self.is_ancestor_mat[self.children_idxs[tree_index]], self.class_features.detach())
            div_n = torch.sum(self.is_ancestor_mat[self.children_idxs[tree_index]], dim=1)
            parent_feature = parent_feature/div_n.view(-1,1)
        knowledge = torch.matmul(p, parent_feature)
        #### refine feature ####
        # attention selector -----------------------------------------
        attention = torch.clamp(torch.tanh(feature+knowledge), min=0)
        refined_feature = feature + attention*knowledge
        #-------------------------------------------------------------

        #### feature calibration ####
        refined_feature = self.alpha*torch.mul(max_score.view(-1,1), refined_feature)
        if tree_index==0:
            refined_feature = self.root_encoder(refined_feature)
        final_dist = self.final_compress[tree_index](refined_feature)
        return first_dist, final_dist


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=False):
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        assert union_features != None 

        # object classes embedding
        if self.use_bias:
            freq_emb = self.freq_bias.index_with_labels(pair_pred.long())

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)
        if self.feat_mode == "concat":
            feature = torch.cat([post_ctx_rep+union_features, freq_emb], dim=-1)
        elif self.feat_mode == "basic":
            feature = post_ctx_rep * union_features

        first_dists = []
        final_dists = []

        for i in range(self.num_tree):
            first_dist, final_dist = self.knowledge_transfer(feature, i)
            first_dists.append(first_dist)
            if ("basic" in self.feat_mode) and self.use_bias and i!=0:
                # # freq info selector 
                final_dist = final_dist + self.freq_compress[i](freq_emb)
                # freq info extraction 
                # --- if you apply it to root ----
                # anc_mat = self.is_ancestor_mat[self.children_idxs[i]]
                # anc_mat /= torch.sum(anc_mat, dim=1).view(-1,1)
                # final_dist = final_dist + torch.matmul(freq_emb, torch.t(anc_mat))
                # --- else ----
                # final_dist = final_dist + torch.matmul(freq_emb, torch.t(self.is_ancestor_mat[self.children_idxs[i]])) #階層分類だと平均とる必要あり, クラスター選択タイプならいらない
            final_dists.append(final_dist)

        final_dists_list = [(final_dists[i].split(num_rels, dim=0)) for i in range(self.num_tree)]

        # TODO code for vstree binary loss
        add_losses = {}
        if self.training:
            rel_labels = cat(rel_labels, dim=0)
            feature_detached = feature.detach()
            if self.feature_loss == "mse":
                add_losses["feature"] = self.mseloss(self.class_features[rel_labels], feature_detached)
            
            elif self.feature_loss == "margin":
                batch_size = feature_detached.size(0)
                # attract loss
                counts = feature_detached.new_ones(self.num_rel_cls)
                counts = counts.scatter_add_(0, rel_labels, counts.new_ones(batch_size))
                class_feature_batch = self.class_features.index_select(0,rel_labels).to(feature_detached.dtype)
                diff = (feature_detached - class_feature_batch).pow(2)/2
                div = counts.index_select(0, rel_labels)
                diff /= div.view(-1,1)
                add_losses["feature attract loss"] = self.weight*diff.sum()/batch_size
                # repel loss
                dist_mat = torch.cdist(feature_detached, self.class_features.to(feature_detached.dtype))
                classes = torch.arange(self.num_rel_cls).long().cuda()
                labels_expand = rel_labels.unsqueeze(1).expand(batch_size, self.num_rel_cls)
                mask = labels_expand.ne(classes.expand(batch_size, self.num_rel_cls)).int()
                distmat_neg = torch.mul(dist_mat, mask)
                # original attract:repel = 1:0.01
                add_losses["feature repel loss"] = self.weight*0.01*torch.clamp(self.margin - distmat_neg.sum()/(batch_size*self.num_rel_cls), 0.0, 1e6)
            
            elif self.feature_loss=="none":
                # moving average
                with torch.no_grad():
                    for i in range(self.num_rel_cls):
                        if len(feature_detached[rel_labels==i]):
                            self.class_features[i] = 0.3*self.class_features[i]+0.7*(feature_detached[rel_labels==i]).mean(dim=0)

            # rel_labels for each cluster
            rel_label_chs = []
            rel_label_chs.append(rel_labels)
            for i in range(1, self.num_tree):
                rel_label_ch = self.global2children[i][rel_labels]
                rel_label_chs.append(rel_label_ch) 
            add_losses['first_prediction'] = torch.zeros([1]).cuda() # combined feature (ctx+union)
            div_n = 0
            for i in range(self.num_tree):
                ####### CROSS ENTROPY LOSS FOR EACH TREE ########
                if (rel_label_chs[i]>=0).sum():
                    div_n += 1
                    add_losses['first_prediction'] += F.cross_entropy(first_dists[i], rel_label_chs[i], ignore_index=-1) 
            # normalization
            if div_n:
                add_losses["first_prediction"] /= div_n

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)
        
        return obj_dist_list, final_dists_list, add_losses

@registry.ROI_RELATION_PREDICTOR.register("CausalPSKTPredictor")
class CausalPSKTPredictor(nn.Module):
    def __init__(self, config, in_channels, taxonomy=None):
        super(CausalPSKTPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.SEPARATE_SPATIAL
        self.transfer = config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.KNOWLEDGE_TRANSFER
        self.feature_loss = config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.FEATURE_LOSS
        self.ctx_feat_path = config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.PRETRAINED_CTX_FEATURE_PATH
        self.vis_feat_path = config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.PRETRAINED_VIS_FEATURE_PATH
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.EFFECT_TYPE
        self.vis_record = False

        if config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                mode = 'predcls'
            else:
                mode = 'sgcls'
        else:
            mode = 'sgdet'
        
        assert in_channels is not None
        num_inputs = in_channels

        # set taxonomy
        self.T = taxonomy
        self.num_tree = self.T["num_tree"]
        self.global2children = torch.from_numpy(self.T["global_to_children"]).cuda()
        self.is_ancestor_mat = torch.from_numpy(self.T["is_ancestor_mat"].astype(config.DTYPE)).cuda()
        self.num_children = self.T["num_children"]
        self.children_idxs = self.T["children"]
        self.children_tree_index = self.T["children_tree_index"]
        if self.num_tree>1:
            self.freq_compress = nn.ModuleList(nn.Linear(self.num_rel_cls, num) for num in self.num_children)

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        self.edge_dim = self.hidden_dim
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                        nn.ReLU(inplace=True),])

        self.ctx_first_compress = nn.ModuleList(nn.Linear(self.pooling_dim, num) for i, num in enumerate(self.num_children))
        self.vis_first_compress = nn.ModuleList(nn.Linear(self.pooling_dim, num) for i, num in enumerate(self.num_children))
        
        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])

        if self.transfer:
            self.ctx_first_compress[0] = nn.Linear(self.pooling_dim, self.num_rel_cls)
            self.vis_first_compress[0] = nn.Linear(self.pooling_dim, self.num_rel_cls)
            self.ctx_final_compress = nn.ModuleList(nn.Linear(self.pooling_dim, num) for num in self.num_children)
            self.vis_final_compress = nn.ModuleList(nn.Linear(self.pooling_dim, num) for num in self.num_children)
            self.class_ctx_features = nn.Parameter(torch.zeros(self.num_rel_cls, self.pooling_dim).to(torch.float), requires_grad=False)
            self.class_vis_features = nn.Parameter(torch.zeros(self.num_rel_cls, self.pooling_dim).to(torch.float), requires_grad=False)
            if self.ctx_feat_path:
                self.class_ctx_features = nn.Parameter(torch.from_numpy(np.load(self.ctx_feat_path, allow_pickle=True).item()["avg_feature"]).to(torch.float).cuda(), requires_grad=False)
            if self.vis_feat_path:
                self.class_vis_features = nn.Parameter(torch.from_numpy(np.load(self.vis_feat_path, allow_pickle=True).item()["avg_feature"]).to(torch.float).cuda(), requires_grad=False)
            if self.feature_loss != "none":
                assert self.feature_loss=="mse" or self.feature_loss=="margin", "Please select 'mse' or 'margin', 'none' for feature loss"
                self.class_ctx_features.requires_grad = True
                self.class_vis_features.requires_grad = True
                self.mseloss = nn.MSELoss()
                if self.feature_loss == "margin":
                    self.margin = 80.0
                    self.weight = 1 if mode == "sgdet" else 0.1
            self.alpha = config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.CALIBRATION_ALPHA
        
        self.layer_initialize()

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

    
    def layer_initialize(self):
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, mode="normal")
        layer_init(self.post_cat[0], mode="xavier")
        if self.spatial_for_vision:
            layer_init(self.spt_emb[0], mode="xavier")
            layer_init(self.spt_emb[2], mode="xavier")
        for i in range(self.num_tree):
            layer_init(self.ctx_first_compress[i], mode="xavier")
            layer_init(self.vis_first_compress[i], mode="xavier")
            if self.num_tree>1:
                layer_init(self.freq_compress[i], mode="xavier")
            if self.transfer:
                layer_init(self.ctx_final_compress[i], mode="xavier")
                layer_init(self.vis_final_compress[i], mode="xavier")

        
    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list

    def refine_feature(self, feature, first_dist, class_features, tree_index):
        p = F.softmax(first_dist, -1)
        max_score, max_idx = p.max(dim=1)
        #### knowledge calculation ####
        if tree_index==0:
            parent_feature = class_features
        else:
            parent_feature = torch.matmul(self.is_ancestor_mat[self.children_idxs[tree_index]], class_features)
            div_n = torch.sum(self.is_ancestor_mat[self.children_idxs[tree_index]], dim=1)
            parent_feature = parent_feature/div_n.view(-1,1)
        knowledge = torch.matmul(p, parent_feature)
        #### refine feature ####
        attention = torch.clamp(torch.tanh(feature+knowledge), min=0)
        refined_feature = feature + attention*knowledge
        #### feature calibration ####
        refined_feature = self.alpha*torch.mul(max_score.view(-1,1), refined_feature)
        return refined_feature
        
    def knowledge_transfer(self, vis_rep, ctx_rep, frq_dist, tree_index):
        if self.num_tree>1:
            frq_dist = self.freq_compress[tree_index](frq_dist)
        vis_first_dist = self.vis_first_compress[tree_index](vis_rep)
        ctx_first_dist = self.ctx_first_compress[tree_index](ctx_rep)
        if not self.transfer:
            return vis_first_dist, ctx_first_dist, None, None, frq_dist

        refined_vis_rep = self.refine_feature(vis_rep, vis_first_dist, self.class_vis_features.detach(), tree_index)
        refined_ctx_rep = self.refine_feature(ctx_rep, ctx_first_dist, self.class_ctx_features.detach(), tree_index)
        vis_final_dist = self.vis_final_compress[tree_index](refined_vis_rep)
        ctx_final_dist = self.ctx_final_compress[tree_index](refined_ctx_rep)
        return vis_first_dist, ctx_first_dist, vis_final_dist, ctx_final_dist, frq_dist
        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats
        
        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)
        
        frq_dist = self.freq_bias.index_with_labels(pair_pred.long())
        
        if self.vis_record:
            self.feature_record(post_ctx_rep, union_features, frq_dist, rel_labels)
            return None, None, {}

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)

        ctx_first_dists = []
        vis_first_dists = []
        ctx_final_dists = []
        vis_final_dists = []
        frq_dists = []

        for i in range(self.num_tree):
            vis_first_dist, ctx_first_dist, vis_final_dist, ctx_final_dist, frq_dist_ = self.knowledge_transfer(union_features, post_ctx_rep, frq_dist, i)
            ctx_first_dists.append(ctx_first_dist)
            vis_first_dists.append(vis_first_dist)
            frq_dists.append(frq_dist_)
            if self.transfer:
                ctx_final_dists.append(ctx_final_dist)
                vis_final_dists.append(vis_final_dist)
        
        final_dists = []
        if self.transfer:
            final_dists = [ctx_final_dists[i]+vis_final_dists[i]+frq_dists[i] for i in range(self.num_tree)]
        else:
            final_dists = [ctx_first_dists[i]+vis_first_dists[i]+frq_dists[i] for i in range(self.num_tree)]

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)
            if self.transfer:
                vis_feat_detached = union_features.detach()
                ctx_feat_detached = post_ctx_rep.detach()
                if self.feature_loss == "mse":
                    add_losses["feature"] = self.mseloss(self.class_vis_features[rel_labels], vis_feat_detached)
                    add_losses["feature"] = self.mseloss(self.class_ctx_features[rel_labels], ctx_feat_detached)
                
                elif self.feature_loss == "margin":
                    batch_size = vis_feat_detached.size(0)
                    # attract loss
                    counts = vis_feat_detached.new_ones(self.num_rel_cls)
                    counts = counts.scatter_add_(0, rel_labels, counts.new_ones(batch_size))
                    class_vis_feature_batch = self.class_vis_features.index_select(0,rel_labels).to(vis_feat_detached.dtype)
                    class_ctx_feature_batch = self.class_ctx_features.index_select(0,rel_labels).to(ctx_feat_detached.dtype)
                    diff_vis = (vis_feat_detached - class_vis_feature_batch).pow(2)/2
                    diff_ctx = (ctx_feat_detached - class_ctx_feature_batch).pow(2)/2
                    div = counts.index_select(0, rel_labels)
                    diff_vis /= div.view(-1,1)
                    diff_ctx /= div.view(-1,1)
                    add_losses["feature attract loss"] = self.weight*(diff_vis.sum()+diff_ctx.sum())/batch_size
                    # repel loss
                    vis_dist_mat = torch.cdist(vis_feat_detached, self.class_vis_features.to(vis_feat_detached.dtype))
                    ctx_dist_mat = torch.cdist(ctx_feat_detached, self.class_ctx_features.to(ctx_feat_detached.dtype))
                    classes = torch.arange(self.num_rel_cls).long().cuda()
                    labels_expand = rel_labels.unsqueeze(1).expand(batch_size, self.num_rel_cls)
                    mask = labels_expand.ne(classes.expand(batch_size, self.num_rel_cls)).int()
                    vis_distmat_neg = torch.mul(vis_dist_mat, mask)
                    ctx_distmat_neg = torch.mul(ctx_dist_mat, mask)
                    # original attract:repel = 1:0.01
                    add_losses["feature repel loss"] = self.weight*0.01*torch.clamp(self.margin - (vis_distmat_neg.sum()+ctx_distmat_neg.sum())/(batch_size*self.num_rel_cls), 0.0, 1e6)
                
                elif self.feature_loss=="none":
                    # moving average
                    with torch.no_grad():
                        for i in range(self.num_rel_cls):
                            if i in rel_labels:
                                self.class_vis_features[i] = 0.3*self.class_vis_features[i]+0.7*(vis_feat_detached[rel_labels==i]).mean(dim=0)
                                self.class_ctx_features[i] = 0.3*self.class_ctx_features[i]+0.7*(ctx_feat_detached[rel_labels==i]).mean(dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            rel_label_chs = []
            for i in range(self.num_tree):
                rel_label_ch = self.global2children[i][rel_labels]
                rel_label_chs.append(rel_label_ch)
            
            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = torch.zeros([1]).cuda()
            add_losses['auxiliary_vis'] = torch.zeros([1]).cuda()
            add_losses['auxiliary_frq'] = torch.zeros([1]).cuda()
            div_n = 0
            for i in range(self.num_tree):
                if (rel_label_chs[i]>=0).sum()>0:
                    div_n += 1
                    add_losses["auxiliary_frq"] += F.cross_entropy(frq_dists[i], rel_label_chs[i], ignore_index=-1)
                    if i==0 and self.transfer:
                        add_losses["auxiliary_ctx"] += F.cross_entropy(ctx_first_dists[i], rel_labels)
                        add_losses["auxiliary_vis"] += F.cross_entropy(vis_first_dists[i], rel_labels)
                    else:
                        add_losses["auxiliary_ctx"] += F.cross_entropy(ctx_first_dists[i], rel_label_chs[i], ignore_index=-1)
                        add_losses["auxiliary_vis"] += F.cross_entropy(vis_first_dists[i], rel_label_chs[i], ignore_index=-1)
                    if self.transfer:
                        add_losses["auxiliary_ctx"] += F.cross_entropy(ctx_final_dists[i], rel_label_chs[i], ignore_index=-1)
                        add_losses["auxiliary_vis"] += F.cross_entropy(vis_final_dists[i], rel_label_chs[i], ignore_index=-1)
            # normalization
            if div_n:
                add_losses["auxiliary_ctx"] /= div_n
                add_losses["auxiliary_vis"] /= div_n
                add_losses["auxiliary_frq"] /= div_n

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep  
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type != "none":
                frq_dist = self.freq_bias.index_with_probability(pair_obj_probs)
                avg_frq_dist = self.freq_bias.index_with_probability(avg_frq_rep)
                for i in range(self.num_tree):
                    if self.num_tree>1 and i==0:
                        continue
                    avg_ctx_dist = self.ctx_first_compress[i](avg_ctx_rep)
                    if self.effect_type == 'TDE':   # TDE of CTX
                        if self.transfer:
                            final_dists[i] = (vis_final_dists[i]+ctx_final_dists[i]+frq_dists[i]) - (vis_final_dists[i]+avg_ctx_dist+frq_dists[i])
                        else:
                            final_dists[i] = (vis_first_dists[i]+ctx_first_dists[i]+frq_dists[i]) - (vis_first_dists[i]+avg_ctx_dist+frq_dists[i])
                    elif self.effect_type == 'NIE': # NIE of FRQ
                        if i>0:
                            avg_frq_dist_ = self.freq_compress[i](avg_frq_dist)
                        if self.transfer:
                            final_dists[i] = (vis_final_dists[i]+avg_ctx_dist+frq_dists[i]) - (vis_final_dists[i]+avg_ctx_dist+avg_frq_dist_)
                        else:
                            final_dists[i] = (vis_first_dists[i]+avg_ctx_dist+frq_dists[i]) - (vis_first_dists[i]+avg_ctx_dist+avg_frq_dist_)
                    elif self.effect_type == 'TE':  # Total Effect
                        if i>0:
                            avg_frq_dist_ = self.freq_compress[i](avg_frq_dist)
                        if self.transfer:
                            final_dists[i] = (vis_final_dists[i]+ctx_final_dists[i]+frq_dists[i]) - (vis_final_dists[i]+avg_ctx_dist+avg_frq_dist_)
                        else:
                            final_dists[i] = (vis_first_dists[i]+ctx_first_dists[i]+frq_dists[i]) - (vis_first_dists[i]+avg_ctx_dist+avg_frq_dist_)
            else:
                assert self.effect_type == 'none'
                pass

        final_dist_list = [final_dist.split(num_rels, dim=0) for final_dist in final_dists]

        return obj_dist_list, final_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder
    
    def feature_record(self, ctx_rep, vis_rep, frq_dist, rel_labels):
        """
        Update feature mean, sum, squared sum
        """
        rel_labels = cat(rel_labels, dim=0).to('cpu').numpy()
        ctx_rep = ctx_rep.detach().to('cpu').numpy()
        vis_rep = vis_rep.detach().to('cpu').numpy()
        frq_dist = frq_dist.detach().to('cpu').numpy()
        for i in range(self.num_rel_cls):
            if len(ctx_rep[rel_labels==i]):
                self.ctx["avg_feature"][i] = 0.3*self.ctx["avg_feature"][i]+0.7*(ctx_rep[rel_labels==i]).mean(axis=0)
                self.vis["avg_feature"][i] = 0.3*self.vis["avg_feature"][i]+0.7*(vis_rep[rel_labels==i]).mean(axis=0)
                self.frq["avg_feature"][i] = 0.3*self.frq["avg_feature"][i]+0.7*(frq_dist[rel_labels==i]).mean(axis=0)


def make_roi_relation_predictor(cfg, in_channels, taxonomy=None):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    if taxonomy!=None:
        return func(cfg, in_channels, taxonomy)
    else:
        return func(cfg,in_channels)
