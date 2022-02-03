import torch
import json, pathlib, os
import numpy as np

def set_groups(config):
    groups_path = config.MODEL.ROI_RELATION_HEAD.MINIGROUP.GROUPS_PATH   
    json_open = open(groups_path, "r")
    groups_json = json.load(json_open)
    num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

    pred2idx = groups_json["predicate_to_idx"]
    pred2idx["bg"] = 0

    num_groups = len(groups_json["groups"])
    groups = []
    for i, num in enumerate(groups_json["groups"]):
        classes = []
        if i == 0:
            classes.append(0)
        for clas in groups_json["groups"][num]:
            classes.append(pred2idx[clas])
        groups.append(np.array(classes))
    
    num_classes = [len(classes) for classes in groups]

    # index mapping list for each tree
    global_to_children = -np.ones([num_groups, num_rel_cls], dtype=int)
    for m in range(num_groups):
        for i_ch, ch in enumerate(groups[m]):
            global_to_children[m][ch] = i_ch

    G =  {
        "groups": groups, "num_groups": num_groups, "num_classes": num_classes, "global_to_children": global_to_children
    }

    # save dict as npy fle
    p = pathlib.PurePath(groups_path)
    savepath = os.path.join(p.parent, p.stem+".npy")
    if not os.path.exists(savepath):
        np.save(savepath, G)

    return G