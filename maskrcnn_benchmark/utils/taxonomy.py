import torch
import logging
import json, pathlib, os
import numpy as np
# from scipy.cluster import hierarchy

logger = logging.getLogger("maskrcnn_benchmark").getChild("predictor")
# def clustering(features, labels, num_cluster, output_path=None):
#     cluster_json = {}
#     cluster_json["children"] = {}
#     cluster_json["children"]["root"] = [f"parent{i}" for i in range(1, num_cluster+1)]
#     z = hierarchy.linkage(features, method="ward", metric="euclidean")
#     cluster = hierarchy.fcluster(z, t=num_cluster, criterion="maxclust")
#     for i, par in enumerate(cluster_json["children"]["root"]):
#         cluster_json["children"][par] = labels[cluster==i+1].tolist()
#     if output_path:
#         with open(output_path, "w") as f:
#             json.dump(cluster_json, f)
#     return cluster_json


def descend(ancestors, cur_idx, is_ancestor_mat, children_tree_index, children):
    # this is internal node, not leaf target label
    if children_tree_index[cur_idx]>=0:
        ancestors = ancestors + [cur_idx]
        tree_index = children_tree_index[cur_idx]
        for ch in children[tree_index]:
            is_ancestor_mat = descend(ancestors, ch, is_ancestor_mat, children_tree_index, children)
        for ancestor in ancestors:
            no_par = children_tree_index[children[tree_index]]<0
            is_ancestor_mat[ancestor][children[tree_index][no_par]] = True
    # leaf target label
    else:
        is_ancestor_mat[cur_idx][cur_idx] = True
    return is_ancestor_mat

def build_taxonomy(config):
    num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
    original_num_rel_cls = num_rel_cls

    #TODO get dataset info from DatasetCatalog
    """ please change this path to your file"""
    vg_info_path = "datasets/vg/VG-SGG-dicts-with-attri.json"
    json_open = open(vg_info_path, "r")
    info = json.load(json_open)
    pred2idx = info["predicate_to_idx"]
    idx2pred = info["idx_to_predicate"]

    # from_ckpt = False
    # pretrained = torch.load(config.MODEL.PRETRAINED_DETECTOR_CKPT, map_location=torch.device("cpu"))
    # class_feat_param = "roi_heads.relation.predictor.class_features"
    # if class_feat_param in pretrained["model"].keys():
    #     from_ckpt = True
    #     class_features = pretrained["model"][class_feat_param][1:] # remove background 0
    #     labels = np.array([idx2pred[str(l)] for l in range(1,num_rel_cls)], dtype=str)
    #     num_cluster = 2
    #     json_path = os.path.join(config.OUTPUT_DIR, "cluster.json")
    # else:
    #     json_path = config.MODEL.ROI_RELATION_HEAD.HIERARCHICALKT.TAXONOMY_PATH 
    if config.MODEL.ROI_RELATION_HEAD.PREDICTOR == "PSKTPredictor":
        json_path = config.MODEL.ROI_RELATION_HEAD.PSKT.TAXONOMY_PATH
    elif config.MODEL.ROI_RELATION_HEAD.PREDICTOR == "PSKTAllPredictor":
        json_path = config.MODEL.ROI_RELATION_HEAD.PSKTALL.TAXONOMY_PATH
    elif config.MODEL.ROI_RELATION_HEAD.PREDICTOR == "PSKTRootAllPredictor":
        json_path = config.MODEL.ROI_RELATION_HEAD.PSKTROOTALL.TAXONOMY_PATH
    elif config.MODEL.ROI_RELATION_HEAD.PREDICTOR == "CausalPSKTPredictor":
        json_path = config.MODEL.ROI_RELATION_HEAD.CAUSALPSKT.TAXONOMY_PATH

    assert json_path!="", "Invalid taxonmy path."
    p = pathlib.PurePath(json_path)
    npy_path = os.path.join(p.parent, p.stem+".npy")
    if os.path.exists(npy_path):
        T = np.load(npy_path, allow_pickle=True)
        return T.item()
    
    # if from_ckpt:
    #     taxonomy_json = clustering(class_features, labels, num_cluster, output_path=json_path)
    # else:
    #     json_open = open(json_path, "r")
    #     taxonomy_json = json.load(json_open)
    
    json_open = open(json_path, "r")
    taxonomy_json = json.load(json_open)

    pred2idx["root"] = -1
    pred2idx["bg"] = 0
    # parents should be super-class
    for i, par in enumerate(taxonomy_json["children"]):
        if i==0: continue # just ignore when parent is root
        pred2idx[par] = num_rel_cls
        num_rel_cls += 1

    num_tree = len(taxonomy_json["children"])
    # indexes in each tree (#tree, #chidlren in each tree)
    children = []
    # parent indexes in each tree (#tree)
    parents = []
    # first index of each tree (#tree+1)
    ch_slice = [0]
    # the index of children tree if any, otherwise just -1 (#nodes)
    children_tree_index = -np.ones(num_rel_cls, dtype=int)
    # parent tree index for each internal_nodes (#tree)
    parent_tree_index = np.zeros(num_tree, dtype=int) 
    # just confirm if all predicates appear
    checked = np.zeros(num_rel_cls, dtype=bool)
    checked[0] = True
    for i, par in enumerate(taxonomy_json["children"]):
        try:
            par_index = pred2idx[par]
        except ValueError:
            raise ValueError(f"Invalid predicate '{par}' in taxonomy json.")
        assert par_index not in parents # avoid overlapping
        child = []
        if par == "root":
            child.append(pred2idx["bg"]) # add index for bg
        else:
            children_tree_index[par_index] = i
        parents.append(par_index)
        for ch in taxonomy_json["children"][par]:
            try:
                ch_idx = pred2idx[ch]
            except ValueError:
                raise ValueError(f"Invalid predicate '{ch}' in taxonomy json.")
            if checked[ch_idx]:
                raise ValueError(f"{ch} appears twice.")
            child.append(ch_idx)
            checked[ch_idx] = True
        children.append(np.array(child))
        ch_slice.append(ch_slice[-1]+len(child))
    
    assert all(checked==True) # confirm all predicates appear in json
    num_children = [len(child) for child in children]

    for i, child in enumerate(children):
        for ch in child:
            tree_idx = children_tree_index[ch]
            if tree_idx != -1:
                parent_tree_index[tree_idx] = i 

    is_ancestor_mat = np.zeros([num_rel_cls, original_num_rel_cls], dtype=bool)
    for ch in children[0]:
        is_ancestor_mat = descend([], ch, is_ancestor_mat, children_tree_index, children)
    
    labels_ch = np.ones([original_num_rel_cls, ch_slice[-1]],dtype=int)
    for m in range(num_tree):
        for i_ch, ch in enumerate(children[m]):
            labels_ch[:, ch_slice[m]+i_ch] = i_ch*is_ancestor_mat[ch].astype(int) - (~is_ancestor_mat[ch]).astype(int)

    # index mapping list for each tree
    global_to_children = -np.ones([num_tree, original_num_rel_cls], dtype=int)
    for m in range(num_tree):
        for i_ch, ch in enumerate(children[m]):
            global_to_children[m][is_ancestor_mat[ch]] = i_ch


    T =  {
        "children" : children, "parents" : parents,
        "children_tree_index": children_tree_index,
        "parent_tree_index": parent_tree_index,
        "global_to_children": global_to_children, 
        "ch_slice": ch_slice,
        "num_children": num_children,
        "num_tree": num_tree,
        "num_node": num_rel_cls,
        "labels_ch": labels_ch,
        "is_ancestor_mat": is_ancestor_mat
    }

    # save dict as npy fle
    np.save(npy_path, T)

    return T