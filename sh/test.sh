#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 \
tools/relation_test_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICTOR  KnowledgeTransferPredictor \
MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM  512 \
MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.CONTEXT_LAYER motifs \
MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.FEAT_MODE concat \
MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.KNOWLEDGE_TRANSFER True \
TEST.IMS_PER_BATCH 1 \
DTYPE "float16" \
OUTPUT_DIR /home/miskai/デスクトップ/related-work/scene-graph-benchmark/SGG-TD2/checkpoints/knowledgetrans-motifs-predcls-exmp/RELU/concat_margin40_from_record_mse_percls_avg_cls0.1
# MODEL.ROI_RELATION_HEAD.PSKT.TAXONOMY_PATH datasets/vg/predcls_clusters/motifs/2cluster.json