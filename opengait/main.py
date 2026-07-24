import argparse
import os

import torch
import torch.nn as nn

from opengait.modeling import models
from opengait.utils import (
    config_loader,
    get_ddp_module,
    get_msg_mgr,
    init_seeds,
    params_count,
)


os.environ["OMP_NUM_THREADS"] = "1"


parser = argparse.ArgumentParser(
    description="Unified train/evaluation entry for ReFuseGait."
)
parser.add_argument(
    "--local_rank",
    "--local-rank",
    type=int,
    default=0,
)
parser.add_argument(
    "--cfgs",
    type=str,
    required=True,
    help="Path to the YAML configuration.",
)
parser.add_argument(
    "--phase",
    choices=["train", "test"],
    required=True,
)
parser.add_argument(
    "--iter",
    default=0,
    help="Iteration to restore from the configured output directory.",
)
parser.add_argument(
    "--log_to_file",
    action="store_true",
)
opt = parser.parse_args()


def configure_test_initialization(cfgs):
    """Configure train-style initialization for standalone evaluation.

    ReFuseGait checkpoints contain the trainable model state, while the frozen
    pretrained event/image components are reconstructed during model
    initialization. To reproduce evaluation performed during training, a
    standalone test therefore initializes the model with training=True and
    switches it to evaluation only after checkpoint restoration.
    """

    evaluator_cfg = cfgs["evaluator_cfg"]
    trainer_cfg = cfgs["trainer_cfg"]

    trainer_cfg["restore_hint"] = evaluator_cfg["restore_hint"]
    trainer_cfg["restore_ckpt_strict"] = evaluator_cfg.get(
        "restore_ckpt_strict",
        True,
    )

    # Optimizer and scheduler states are irrelevant for evaluation.
    trainer_cfg["optimizer_reset"] = True
    trainer_cfg["scheduler_reset"] = True
    trainer_cfg["with_test"] = False

    # Keep standalone test logs separated from training logs.
    trainer_cfg["save_name"] = evaluator_cfg["save_name"]


def initialization(cfgs, phase):
    msg_mgr = get_msg_mgr()

    # Both training and standalone testing require train-style model
    # initialization. Testing switches to eval mode in run_model().
    engine_cfg = cfgs["trainer_cfg"]

    output_path = os.path.join(
        "output",
        cfgs["data_cfg"]["dataset_name"],
        cfgs["model_cfg"]["model"],
        engine_cfg["save_name"],
    )

    restore_iteration = (
        engine_cfg["restore_hint"]
        if isinstance(engine_cfg["restore_hint"], int)
        else 0
    )

    msg_mgr.init_manager(
        output_path,
        opt.log_to_file,
        engine_cfg["log_iter"],
        restore_iteration,
    )

    if phase == "test":
        msg_mgr.log_info(
            "[UNIFIED-TEST] train-style initialization, "
            "then standalone evaluation."
        )

    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def prepare_standalone_evaluation(model, cfgs):
    """Attach evaluation-only state after train-style initialization."""

    model.eval()

    eval_model = (
        model.module
        if hasattr(model, "module")
        else model
    )

    if not hasattr(eval_model, "test_loader"):
        eval_model.test_loader = eval_model.get_loader(
            cfgs["data_cfg"],
            train=False,
        )

    if not hasattr(eval_model, "evaluator_trfs"):
        from opengait.data.transform import get_transform

        eval_model.evaluator_trfs = get_transform(
            cfgs["evaluator_cfg"]["transform"]
        )


def run_model(cfgs, phase):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs["model_cfg"]

    msg_mgr.log_info(model_cfg)

    Model = getattr(
        models,
        model_cfg["model"],
    )

    # Intentionally use training=True for both phases. This is required to
    # reproduce the initialization used by evaluation during training.
    model = Model(
        cfgs,
        training=True,
    )

    if cfgs["trainer_cfg"]["sync_BN"]:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            model
        )

    if cfgs["trainer_cfg"]["fix_BN"]:
        model.fix_BN()

    model = get_ddp_module(
        model,
        cfgs["trainer_cfg"]["find_unused_parameters"],
    )

    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if phase == "train":
        Model.run_train(model)
        return

    prepare_standalone_evaluation(
        model,
        cfgs,
    )
    Model.run_test(model)


if __name__ == "__main__":
    torch.distributed.init_process_group(
        "nccl",
        init_method="env://",
    )

    world_size = torch.distributed.get_world_size()
    cuda_count = torch.cuda.device_count()

    if world_size != cuda_count:
        raise ValueError(
            "Expected world size to equal visible GPU count: "
            f"world_size={world_size}, visible_gpus={cuda_count}"
        )

    cfgs = config_loader(opt.cfgs)

    if opt.iter != 0:
        iteration = int(opt.iter)
        cfgs["evaluator_cfg"]["restore_hint"] = iteration
        cfgs["trainer_cfg"]["restore_hint"] = iteration

    if opt.phase == "test":
        configure_test_initialization(cfgs)

    initialization(
        cfgs,
        opt.phase,
    )
    run_model(
        cfgs,
        opt.phase,
    )
