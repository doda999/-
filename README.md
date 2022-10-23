# SGG-PS
## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.  
Our codebase uses PyTorch 1.8.0 with CUDA 11.1.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Pre-Trained Models

Here is the checkpoints of our models on PredCls (batch size 12) / SGDet, SGCls (batch size 4): [link](https://drive.google.com/drive/folders/1wADFoIaFJEjTq4rfw6vohNsZERQTHL6t?usp=sharing).

Corresponding Results.
Models | mR@20 | mR@50 | mR@100 | zR@20 | zR@50 | zR@100
-- | -- | -- | -- | -- | -- | -- 
MOTIFS-Predcls-ours | 15.84 | 19.75 | 21.28 | 5.51 | 8.95 | 10.6
MOTIFS-SGCls-ours | 12.35 | 15.53 | 16.73 | 1.87 | 3.11 | 3.77
MOTIFS-SGDet-ours | 7.22 | 10.05 | 12.42 | 0.26 | 0.66 | 1.38

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
+ ```CausalPSKTPredictor```

Among them, our implementations is ```CausalPSKTPredictor```. Important options for our models are following:  
+ ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.FINETUNE_FOR_RELATION``` is whether you freeze everything except relation classifiers while training. We set it to ```TRUE``` except when training a baseline model.
+ ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.CONTEXT_LAYER``` is used to select SGG backbone models : ```motifs``` or ```vctree```.  
+ ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.KNOWLEDGE_TRANSFER``` is whether you use the knowledge transfer module while training.  
+ ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.CALIBRATION_ALPHA``` is a constant value for the final scaling in knowledge transfer.
+ ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.TAXONOMY_PATH``` is used to set taxonomy path to branch the process.
+ ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.PRETRAINED_CTX_FEATURE_PATH```, ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.PRETRAINED_VIS_FEATURE_PATH``` are average relation cntexts and union features to initialize memory features.
+ ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.FEATURE_LOSS_WEIGHT``` is a weight for feature loss for knowledge transfer.  
+ ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.EFFECT_ANALYSIS``` is whether you apply causal effect analysis in inference. 
+ ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.EFFECT_TYPE``` is used to set the causal effect analysis type in inference: ```none```, ```TDE```, ```TE```. Note that this should be ```none``` during training. Refer to [this paper](https://arxiv.org/abs/2002.11949) for the details.

To train our model, we need to do two things in advance:
+ Calculate average features to initialize memory features.
+ Cluster predicates with average probability distributions.

You can complete these in two ways. The easiest way is to use pretrained taxonomies and features in [this folder](datasets/vg). Using them, train our model by running ```bash sh/train_causalpskt.sh```.

The other way is to train a baseline and calculate average distributions/features by yourself. Follow the instructions below.

Firstly, train ```CausalPSKTPredictor``` with the options as follows:
```bash
MODEL.PRETRAINED_DETECTOR_CKPT [path to pretrained Faster R-CNN backbone checkpoint]
MODEL.ROI_RELATION_HEAD.FINETUNE_FOR_RELATION False 
MODEL.ROI_RELATION_HEAD.CAUSALPSKT.KNOWLEDGE_TRANSFER False 
MODEL.ROI_RELATION_HEAD.CAUSALPSKT.TAXONOMY_PATH datasets/vg/taxonomy/flat.json
```

After training the baseline, run ```sh/record.sh```. In output, ```rel.npy``` is for the average probability distributions. ```ctx.npy``` and ```vis.npy``` corresopnd to the average relation contexts and union features, respectively. For clustering, taxonomies can be manually constructed with [this code](analysis/clustering.ipynb) from ```rel.npy```. Then run ```bash sh/train_causalpskt.sh``` with your own path.

</br>

**Note**: Since we are still working on improving the vctree based model, the result of our vctree model will be lower than baseline. Therefore, we recommend to set ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.CONTEXT_LAYER``` to ```motifs```.

## Evaluation
To evaluate your model, run ```bash sh/test.sh```. Note that ```MODEL.ROI_RELATION_HEAD.CAUSALPSKT.TAXONOMY_PATH``` should be the same as the taxonomy path you set in training. 

## Acknowledgement
This repository is created based on the scene graph benchmark codebase by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
