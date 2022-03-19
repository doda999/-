CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10026 --nproc_per_node=1 \
tools/feat_record.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICTOR  KnowledgeTransferPredictor \
MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM  512 \
MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.CONTEXT_LAYER motifs \
MODEL.ROI_RELATION_HEAD.KNOWLEDGETRANS.KNOWLEDGE_TRANSFER False \
TEST.IMS_PER_BATCH 1 \
OUTPUT_DIR /home/miskai/デスクトップ/related-work/scene-graph-benchmark/SGG-TD2/checkpoints/motifs-predcls-exmp/RELU/concat-less-lr0.001 