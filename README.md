# SGG-PS

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.  
Our codebase uses PyTorch 1.8.0 with CUDA 11.1.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Pre-Trained Models

Here is the checkpoints of our models on SGDet (batch size 4)/SGCls, PredCls (batch size 12): (Now preparing sharing links...)  
**Note**: We tranined with only 1 NVIDIA A100 GPU (40GB) and we're not yet sure whether this code works with multiple GPUs. But, we're not planning on fixing this currently. Feel free to open PR if you want to help the code update.

Corresponding Results.
Models |  R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 | zR@20 | zR@50 | zR@100
-- | -- | -- | -- | -- | -- | -- | -- | -- | -- 
MOTIFS-Predcls-ours   | 50.62 | 57.27 | 59.24 | 12.92 | 17.22 | 19.06 | 2.02 | 4.56 | 6.36
MOTIFS-SGCls-ours    | 32.11 | 35.53 | 36.47 | 8.34 | 10.69 | 11.7 | 0.45 | 0.95 | 1.47
MOTIFS-SGDet-ours  | 20.9 | 27.2 | 31.33 | 4.85 | 6.90 | 8.44 | 0.03 | 0.21 | 0.51

As for Faster R-CNN backbone model, you can gain pretrained checkpoint from [here](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch#pretrained-models).

## Training
Basically most options are explained in [this repository](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). 

There are three tasks.  
For **Predicate Classification (PredCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```
For **Scene Graph Classification (SGCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
For **Scene Graph Detection (SGDet)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

### Models
Various SGG models are implemented in the file ```roi_heads/relation_head/roi_relation_predictors.py```. You can select relation predictors by changing ```MODEL.ROI_RELATION_HEAD.PREDICTOR``` to one of the available models  
+ ```TransformerPredictor```
+ ```IMPPredictor```
+ ```MotifPredictor```
+ ```VCTreePredictor```
+ ```CausalAnalysisPredictor```
+ ```KnowledgeTransferPredictor```
+ ```PSKTPredictor```  

Among them, our implementations are ```KnowledgeTransferPredictor``` and ```PSKTPredictor```. ```KnowledgeTransferPredictor``` is reimplementation of [this paper](https://arxiv.org/abs/2006.07585) to compare with others on the same codebase. ```PSKTPredictor``` is our model implementation.  

<details><summary>For KnowledgeTransferPredictor</summary>

Run  
```bash
bash sh/motifs_train_kt.sh
```  
+ ```MODEL.ROI_RELATION_HEAD.FINETUNE_FOR_RELATION``` is whether you freeze everything except relation classifiers while training.  
+ ```MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.KNOWLEDGE_TRANSFER``` is whether you use knowledge transfer while training.  
+ ```MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.CONTEXT_LAYER``` is used to select SGG base models : ```motifs``` or ```vctree```.  
+ ```MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.FEATURE_LOSS_WEIGHT``` is weight for feature loss for knowledge transfer.  

Our baseline implementation is slightly different from original SGG base model such as ```MotifPredictor``` and ```VcTreePredictor```.  
To train our baseline, train ```KnowledgeTransferPredictor``` with :
```bash
MODEL.ROI_RELATION_HEAD.FINETUNE_FOR_RELATION False MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.KNOWLEDGE_TRANSFER False
```


To train a predictor using knoweldge transfer, you need a baseline checkpoint and average feature for each relation class. Average feature data is used to initialize feature centroids. You can use ```sh/feat_record.sh``` to calculate average, but it's easier to use precalculated feature [(here)](datasets/vg/initial_feature).  
After obtaining initial feature, train ```KnowledgeTransferPredictor``` with :
```bash
MODEL.ROI_RELATION_HEAD.FINETUNE_FOR_RELATION True MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.KNOWLEDGE_TRANSFER False MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.INITIAL_FEATURE_PATH [your initial feature path] MODEL.PRETRAINED_DETECTOR_CKPT [your baseline model]
```
</details>


<details><summary>For PSKTPredictor</summary>

**Note**: Firstly you need to train a baseline model to follow our steps completely. To train the baseline, refer the section above.

With your basline checkpoint, run
```bash
bash sh/motifs_train_kt.sh
```  
+ ```MODEL.ROI_RELATION_HEAD.PSKT.TAXONOMY_PATH``` is used to set taxonomy path. Taxonomy can be manually constructed with [this code](analysis/clustering.ipynb) from [average feature data](datasets/vg/initial_feature), but we already prepared the taxonomy for the data in [this folder](datasets/vg/taxonomy). 

Other options are same as the options with same name for ```KnowledgeTransferPredictor```, so refer the section above.  

```MODEL.ROI_RELATION_HEAD.FINETUNE_FOR_RELATION``` should be True, but if you want to train whole model jointly (it's not our implementation), you can turn it to False.   
if ```MODEL.ROI_RELATION_HEAD.FINETUNE_FOR_RELATION``` is True, you can apply knowledge transfer in addition to the taxonomy based on predicate similarities. 
</details>
<br />

**Note**: Since we are still working on improving the vctree based model, the result of our vctree model will be lower than baseline. Therefore, we recommend set ```MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.CONTEXT_LAYER``` or ```MODEL.ROI_RELATION_HEAD.PSKT.CONTEXT_LAYER``` to ```motifs```.
