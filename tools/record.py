import argparse
from statistics import mode
from maskrcnn_benchmark.utils.timer import Timer
import os
import numpy as np
from torch.utils import data
from tqdm import tqdm

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.timer import Timer, get_time_str

try:
    from apex import amp
except ImportError:
    raise ImportError("Use APEX precision via apex.amp.")

def record(model, data_loader, predictor, output_folder, device, timer=None, logger=None):
    # save file
    vis_savefile = os.path.join(output_folder,"vis.npy")
    ctx_savefile = os.path.join(output_folder,"ctx.npy")
    frq_savefile = os.path.join(output_folder,"frq.npy")
    rel_savefile = os.path.join(output_folder, "rel.npy")
    model.roi_heads.relation.predictor.vis = {"avg_feature": np.zeros((model.roi_heads.relation.predictor.num_rel_cls, model.roi_heads.relation.predictor.pooling_dim))}
    model.roi_heads.relation.predictor.ctx = {"avg_feature": np.zeros((model.roi_heads.relation.predictor.num_rel_cls, model.roi_heads.relation.predictor.pooling_dim))}
    model.roi_heads.relation.predictor.frq = {"avg_feature": np.zeros((model.roi_heads.relation.predictor.num_rel_cls, model.roi_heads.relation.predictor.num_rel_cls))}
    model.roi_heads.relation.predictor.rel = {"avg_feature": np.zeros((model.roi_heads.relation.predictor.num_rel_cls, model.roi_heads.relation.predictor.num_rel_cls))}

    model.eval()
    cpu_device = torch.device("cpu")
    torch.cuda.empty_cache()
    for itr, batch in enumerate(tqdm(data_loader)):
        if itr == 20653: continue
        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]
            if timer:
                timer.tic()
            model(images.to(device), targets)
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
    np.save(vis_savefile, model.roi_heads.relation.predictor.vis)
    np.save(ctx_savefile, model.roi_heads.relation.predictor.ctx)
    np.save(frq_savefile, model.roi_heads.relation.predictor.frq)
    np.save(rel_savefile, model.roi_heads.relation.predictor.rel)
    torch.cuda.empty_cache()
    return 

def main():
    parser = argparse. ArgumentParser(description="Pytorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    predictor = cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.freeze()

    assert cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR=="CausalPSKTPredictor"

    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank(), filename="visrecord_log.txt")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.roi_heads.relation.record = True
    model.roi_heads.relation.predictor.record = True
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "record")
        mkdir(output_folder)

    data_loader_train = make_data_loader(cfg, mode="train", is_distributed=distributed, record=True)[0]

    device = torch.device("cuda")
    num_devices = get_world_size()
    dataset = data_loader_train.dataset

    logger.info("Start recording on dataset({} images).".format(len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    # TODO: need new func
    record(model, data_loader_train, predictor, output_folder, device, timer=inference_timer, logger=logger)
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
        total_time_str, total_time * num_devices / len(dataset), num_devices 
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

if __name__ == "__main__":
    main()
    
