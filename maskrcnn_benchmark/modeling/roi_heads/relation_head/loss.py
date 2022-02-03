# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import numpy.random as npr
import logging

from maskrcnn_benchmark.layers import smooth_l1_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.roi_heads.attribute_head import loss
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
        taxonomy=None,
        groups=None,
        is_pcpl=False
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()
        self.logger = logging.getLogger("maskrcnn_benchmark").getChild("loss")

        self.is_hierarchical = taxonomy!=None
        self.is_groupbased = groups!=None
        self.is_pcpl = is_pcpl
        self.rel_loss = nn.CrossEntropyLoss()
        if self.is_hierarchical:
            self.rel_loss = TreeLoss(taxonomy)
        elif self.is_groupbased:
            self.rel_loss = GroupLoss(groups)
        elif self.is_pcpl:
            self.rel_loss = PCPLLoss()
        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits, ind=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        if not (self.is_hierarchical or self.is_groupbased):
          relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        if self.is_pcpl:
            loss_relation = self.rel_loss(relation_logits, rel_labels.long(), ind)
        else:
            loss_relation = self.rel_loss(relation_logits, rel_labels.long())
        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:
            return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss


# fixed version for hierarchical structure
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.ignore_index = ignore_index
 
    def forward(self, input, target):
        target = target.view(-1)
        # mask of samples which should be ignored
        mask = target == self.ignore_index
        # replace -1 with 0 so that target can be used
        target[mask] = 0

        logpt = F.log_softmax(input)
        # The returned tensor has the same number of dimensions as the original tensor (input). 
        # The dimth dimension has the same size as the length of index; other dimensions have the same size as in the original tensor.
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)[~mask]
        # logpt = logpt.index_select(-1, target).diag()
        # logpt = logpt.view(-1)

        # possibility of class at target index 
        pt = logpt.exp()
        
        # for binary classification?
        # logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class HierarchicalLoss(nn.Module):
    def __init__(self, taxonomy, lmbda=1):
        super(HierarchicalLoss, self).__init__()
        self.treeloss = TreeLoss(taxonomy)
        self.stsloss = STSLoss(taxonomy)
        self.lmbda = lmbda
        self.logger = logging.getLogger("maskrcnn_benchmark").getChild("hierarchicalloss")

    def forward(self, input, target):
        loss_relation = torch.zeros([1]).cuda()
        loss_tree = self.treeloss(input, target)
        loss_sts = self.stsloss(input, target)
        loss_relation = loss_tree+self.lmbda*loss_sts
        return loss_relation

class TreeLoss(nn.Module):
    def __init__(self, taxonomy):
        super(TreeLoss, self).__init__()
        self.T = taxonomy
        self.global2children = torch.from_numpy(self.T["global_to_children"]).cuda().to(torch.long)
        self.num_tree = self.T["num_tree"]

        # self.celoss = FocalLoss(gamma=2, ignore_index=-1)
        self.celoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.logger = logging.getLogger("maskrcnn_benchmark").getChild("treeloss")
    def forward(self, input, target):
        treeloss = torch.zeros([1]).cuda()
        div_n = 0
        for i in range(self.num_tree):
            logits = cat(input[i], dim=0)
             ####### CROSS ENTROPY LOSS FOR EACH TREE ########
            target_ch = self.global2children[i][target]
            if (target_ch>=0).sum() > 0:
                div_n += 1
                treeloss += self.celoss(logits, target_ch)
         # normalization per class
        if div_n>0:
            treeloss /= div_n
        return treeloss

# stochastic sampling loss 
class STSLoss(nn.Module):
    def __init__(self, taxonomy, dropout_rate=0.2):
        super(STSLoss, self).__init__()
        self.dropout_rate = dropout_rate
        self.RandomDrop = nn.Dropout(self.dropout_rate)
        self.T = taxonomy
        self.num_tree = self.T["num_tree"]
        self.num_node = self.T["num_node"]
        self.children = self.T["children"]
        self.children_tree_idx = torch.from_numpy(self.T["children_tree_index"])
        self.parent_tree_idx = self.T["parent_tree_index"]
        self.is_ancestor_mat = torch.from_numpy(self.T["is_ancestor_mat"]).float().cuda()
        self.logger = logging.getLogger("maskrcnn_benchmark").getChild("stsloss")

    def forward(self, input, target):
        logits = cat(input[0], dim=0)
        batch_size = logits.shape[0]
        gate = torch.ones(batch_size, self.num_tree).cuda()
        gate = self.RandomDrop(gate)*(1-self.dropout_rate)
        gate[:,0] = 1 # don't drop root

        outs = []
        masks = []
        # which leaf nodes that each node under root has an influence on
        cw = self.is_ancestor_mat[torch.from_numpy(self.children[0])]
        outs.append(torch.matmul(logits, cw))
        for i in range(1, self.num_tree):
            logits = cat(input[i], dim=0)
            par_tree = i
            cw = self.is_ancestor_mat[torch.from_numpy(self.children[i])]
            # if any of the ancestors is dropped out, the node is ignored out as well
            cond_gate = torch.ones([batch_size, 1]).cuda()
            while par_tree != 0:
                cond_gate = torch.mul(cond_gate, gate[:, par_tree].view(-1,1))
                par_tree = self.parent_tree_idx[par_tree]
            outs.append(torch.matmul(logits, cw)*cond_gate)
            mask = torch.zeros([batch_size, self.num_node]).cuda()
            mask[:, torch.from_numpy(self.children[i])] = 1
            masks.append(mask*cond_gate)
        output = torch.clamp(torch.sum(torch.stack(outs), 0), 1e-17, 1)
        out_mask = torch.sum(torch.stack(masks), 0)
        out_mask[:, torch.from_numpy(self.children[0])] = 1
        out_mask = out_mask[:, (self.children_tree_idx==-1).nonzero().flatten()]
        sfmx_base = torch.sum(torch.exp(output)*out_mask, 1)
        gt_z = torch.gather(output, 1, target.view(-1, 1))
        stsloss = torch.mean(-gt_z + torch.log(sfmx_base.view(-1,1)))
        return stsloss

# completely same as TreeLoss
class GroupLoss(nn.Module):
    def __init__(self, groups):
        super(GroupLoss, self).__init__()
        self.G = groups
        self.global2children = torch.from_numpy(self.G["global_to_children"]).cuda().to(torch.long)
        self.num_groups = self.G["num_groups"]

        self.celoss = FocalLoss(gamma=2, ignore_index=-1)
        # self.celoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.logger = logging.getLogger("maskrcnn_benchmark").getChild("grouploss")
    def forward(self, input, target):
        grouploss = torch.zeros([1]).cuda()
        div_n = 0
        for i in range(self.num_groups):
            logits = cat(input[i], dim=0)
             ####### CROSS ENTROPY LOSS FOR EACH TREE ########
            target_ch = self.global2children[i][target]
            if (target_ch>=0).sum() > 0:
                div_n += 1
                grouploss += self.celoss(logits, target_ch)
         # normalization per class
        if div_n>0:
            grouploss /= div_n
        return grouploss

# consider class independence 
class PCPLLoss(nn.Module):
    def __init__(self, size_average=True):
        super(PCPLLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.size_average = size_average
        self.logger = logging.getLogger("maskrcnn_benchmark").getChild("pcplloss")
    def forward(self, input, target, ind):
        # relation class present at the batch
        gt_cls = torch.unique(target)
        weight = ind/torch.sum(ind[gt_cls])
        logpt = F.log_softmax(input)
        logpt = logpt.index_select(-1, target).diag()
        loss = -weight[target]*logpt.view(-1)
        if self.size_average: return loss.mean()
        else: return loss.sum()

def make_roi_relation_loss_evaluator(cfg, taxonomy=None, groups=None, is_pcpl=False):
    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        taxonomy=taxonomy,
        groups=groups,
        is_pcpl=is_pcpl
    )

    return loss_evaluator
