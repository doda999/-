# SGG-PS

## Installation
We use PyTorch 1.8.0 with CUDA 11.1. You can install all required environments at once by running `bash sh/install.sh`

if any conflicts, refer [pytorch official cite](https://pytorch.org/get-started/previous-versions/) and install PyTorch to match your environment.

If you'd rather install step by step, follow [INSTALL.md](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/INSTALL.md).

## Dataset
Check [DATASET.md](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md) for instructions of dataset preprocessing.

## Pre-Trained Models

To train your own models you can obtain the weights for the pretrained detector from [this repository](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Training
We use 1 NVIDIA GPU with 40GB to train our models.  
Here how to train various models under various tasks.
1. SGG tasks  
    * For PredCls set  
    `MODEL.ROI_RELATION_HEAD.USE_GT_BOX True`  
    `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True`
    * For SGCls set  
    `MODEL.ROI_RELATION_HEAD.USE_GT_BOX True`
    `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False`  
    * For SGDet set  
    `MODEL.ROI_RELATION_HEAD.USE_GT_BOX False`  
    `MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False`  
2. SGG models  
You can select various SGG models by changing `MODEL.ROI_RELATION_HEAD.PREDICTOR` option to one of the models below.
    * `MotifPredictor`
    * `VCTreePredictor`
    * `IMPPredictor`
    * `TransformerPredictor`
    * `CausalAnalysisPredictor`
    * `KnowledgeTransferPredictor`  
    * `PSKTPredictor`

    Among them, our implemented predictor is `KnowledgeTransferPredictor` and `PSKTPredictor`. `KnowledgeTransferPredictor` is re-implemented model of [the paper](https://arxiv.org/abs/2006.07585) so that it can be trained in the same codebase as others. `PSKTPredictor` is our predictor model.  


For KnowledeTransferPredictor:  
run `bash sh/motifs_train_kt.sh`  
`MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.KNOWLEDGE_TRANSFER` is whether you apply knowledge transfer. When it's True, `MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.FEATURE_LOSS` can be used to set the type of loss to train feature codewords to transfer. `MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.FEAT_MODE` is used to set how to calculate feature as relation predictor input. 
Though our baseline is based on motifs/vctree, we slightly change the implementations from original ones (`MotifPredictor`, `VCTreePredictor`). To train our baseline, train `KnowledgeTransferPredictor` with `MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.KNOWLEDGE_TRANSFER False` , `MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.FEAT_MODE concat` and `MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.CONTEXT_LAYER motifs`. 

Note that We are now working on improving vctree based models, so we recommend set `MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.CONTEXT_LAYER` or `MODEL.ROI_RELATION_HEAD.PSKT.CONTEXT_LAYER` to `motifs`.

As for other models, please refer [this instructions](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch#predefined-models).








