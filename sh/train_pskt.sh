CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10026 --nproc_per_node=1 \
tools/relation_train_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICTOR  PSKTPredictor \
MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM  512 \
MODEL.ROI_RELATION_HEAD.FINETUNE_FOR_RELATION  True \
MODEL.ROI_RELATION_HEAD.PSKT.CONTEXT_LAYER motifs \
MODEL.ROI_RELATION_HEAD.PSKT.FEATURE_LOSS_WEIGHT 1.0 \
MODEL.ROI_RELATION_HEAD.PSKT.KNOWLEDGE_TRANSFER True \
SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 \
DTYPE "float16" \
SOLVER.BASE_LR 0.001 \
SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 \
MODEL.PRETRAINED_DETECTOR_CKPT /home/miskai/デスクトップ/related-work/scene-graph-benchmark/SGG-TD2/checkpoints/motifs-predcls-exmp/RELU/concat-less-lr0.001/model_0020000.pth \
OUTPUT_DIR /home/miskai/デスクトップ/related-work/scene-graph-benchmark/SGG-PS/checkpoints/for_debug \
MODEL.ROI_RELATION_HEAD.PSKT.TAXONOMY_PATH datasets/vg/taxonomy/predcls/motifs/2cluster.json \
MODEL.ROI_RELATION_HEAD.PSKT.INITIAL_FEATURE_PATH datasets/vg/initial_feature/predcls/feat.npy


