
from email.policy import default

import torch
import pickle
import yaml
import time
import random
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.nn.modules import loss
from collections import deque
from datetime import datetime, timedelta


from config.config import config
from easydict import EasyDict as edict

from utils.function import plot_tsne_with_centroids
from utils.AverageMeter import AverageMeter

from model.CosineFace import CosFace
from model.ArcFace import ArcFace
from loader.Coord_Dataset_experiments import Coord_Dataset as Coord_Dataset_EXP
import warnings

warnings.filterwarnings(
  "ignore",
  message=(
      r"The epoch parameter in "
      r"`scheduler\.step\(\)` was not necessary.*"
  ),
  category=UserWarning,
  module=r"torch\.optim\.lr_scheduler",
)

def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    import os
    from setproctitle import *

    setproctitle('ExponentialLR-gamma:0.93/256,256')

    fix_seed(config.SEED)

    gen_config('/home/jysuh/PycharmProjects/coord_embedding/coord_embed.yaml')

    NUM_JOINTS = config.DATASET.NUM_JOINTS
    NUM_TOKEN = config.DATASET.NUM_TOKEN

    device = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")

    # [2] Vocab Load
    with open(config.DATASET.VALID_JOINT_VOCAB_PATH, 'rb') as f:
        joint_vocab = pickle.load(f)

    criterion = nn.CrossEntropyLoss().to(device)

    #
    base_token_ids = torch.arange(NUM_TOKEN + NUM_JOINTS, device=device, dtype=torch.long).unsqueeze(0)

    #
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #
    total_configurations = (
            len(config.EXP.FEATURE_MODES)
            * len(config.EXP.EMB_MODE)
            * len(config.EXP.NUM_LAYERS)
            * len(config.EXP.S_RANGE)
            * len(config.EXP.M_RANGE)
    )

    completed_configurations = 0
    experiment_start_time = time.time()

    # 최근 10개 configuration 시간으로 ETA 계산
    recent_configuration_times = deque(maxlen=10)

    print(
        f"Total configurations: "
        f"{total_configurations}"
    )
    #
    for FEATURE_MODE in config.EXP.FEATURE_MODES:
        # [3] Data Loader
        TRAIN_DATA_PATH = f'/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/fitness_data_preprocess/ArcFace Dataset/TRAIN_ArcFace_{FEATURE_MODE}.pkl'
        VALID_DATA_PATH = f'/home/jysuh/PycharmProjects/BERTSUMFORHPE(integrated)/fitness_data_preprocess/ArcFace Dataset/VALID_ArcFace_{FEATURE_MODE}.pkl'
        for EMB_MODE in config.EXP.EMB_MODE:
            #
            train_dataset = Coord_Dataset_EXP(config=config, mode='TRAIN', d_path=TRAIN_DATA_PATH, embedding_mode=EMB_MODE)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=config.TRAIN.SHUFFLE,
                num_workers=config.WORKERS,
                pin_memory=True
            )

            valid_dataset = Coord_Dataset_EXP(config=config, mode='VALID', d_path=VALID_DATA_PATH, embedding_mode=EMB_MODE)

            sample_generator = torch.Generator()
            sample_generator.manual_seed(config.VIS.TSNE_RANDOM_SEED)

            num_pose_samples = min(
                config.VIS.SAMPLES_PER_CLASS,
                len(valid_dataset)
            )

            sample_indices = torch.randperm(
                len(valid_dataset),
                generator=sample_generator
            )[:num_pose_samples].tolist()

            valid_subset = torch.utils.data.Subset(
                valid_dataset,
                sample_indices
            )

            valid_loader = torch.utils.data.DataLoader(
                valid_subset,
                batch_size=config.VALID.BATCH_SIZE,
                shuffle=False,
                num_workers=config.WORKERS,
                pin_memory=True
            )
            for NUM_LAYERS in config.EXP.NUM_LAYERS:
                for S in config.EXP.S_RANGE:
                    for M in config.EXP.M_RANGE:
                        IN_DIM = config.FEATURE_MODES_IN_DIM[FEATURE_MODE] - 1 if EMB_MODE != 'RwID' else config.FEATURE_MODES_IN_DIM[FEATURE_MODE]
                        FILE_NAME = (f'[{EMB_MODE}] '
                            f'LAYERS_NUM:{NUM_LAYERS} '
                            f'IN_DIM:{IN_DIM} '
                            f'DIM:{config.MODEL.OUT_CHANNELS} '
                            f'S:{S} '
                            f'M:{M}')
                        #
                        configuration_start_time = time.time()
                        configuration_index = completed_configurations + 1

                        print(
                            f"\nStarting configuration "
                            f"[{configuration_index}/{total_configurations}]"
                        )
                        print(FILE_NAME)

                        #
                        SAVE_ROOT = f"/home/jysuh/PycharmProjects/coord_embedding/checkpoint/{FEATURE_MODE}/{EMB_MODE}"
                        #
                        fc_metric = ArcFace(num_layer=NUM_LAYERS, in_features=IN_DIM, \
                                            out_features=config.MODEL.OUT_CHANNELS, num_class=NUM_JOINTS + NUM_TOKEN, \
                                            embedding_mode=EMB_MODE, activation=config.TRAIN.ACT, device=device).to(device)
                        fc_metric.train()

                        # Optimizer, Scheduler
                        optimizer = torch.optim.Adam(
                            fc_metric.parameters(),
                            lr=config.TRAIN.LR
                        )

                        cosine_epochs = config.TRAIN.EPOCH - config.TRAIN.WARMUP_EPOCH

                        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                            optimizer,
                            start_factor=0.1,
                            end_factor=1.0,
                            total_iters=config.TRAIN.WARMUP_EPOCH
                        )

                        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=cosine_epochs,
                            eta_min=1e-6
                        )

                        scheduler = torch.optim.lr_scheduler.SequentialLR(
                            optimizer,
                            schedulers=[
                                warmup_scheduler,
                                cosine_scheduler
                            ],
                            milestones=[config.TRAIN.WARMUP_EPOCH]
                        )
                        #
                        best_loss = float("inf")
                        best_epoch = None
                        #
                        for epoch in range(config.TRAIN.EPOCH):
                            batch_time.reset()
                            data_time.reset()
                            losses.reset()
                            #
                            end = time.time()
                            #
                            epoch_start_time = time.time()
                            for i ,(J_coord) in enumerate(train_loader):
                                BS = J_coord.shape[0]
                                #
                                loss = 0
                                #
                                J_coord = J_coord.to(device, non_blocking=True)
                                token_ids = base_token_ids.expand(BS, -1)
                                optimizer.zero_grad()

                                logits, _ = fc_metric(input=J_coord, token_ids=token_ids, use_arcface=True, m=M, s=S)
                                #
                                labels = token_ids.reshape(-1).long()
                                loss = criterion(logits, labels)
                                #
                                loss.backward()
                                losses.update(loss.detach(), J_coord.size(0))
                                #
                                batch_time.update(time.time() - end)
                                end = time.time()
                                #
                                optimizer.step()
                            #
                            #
                            epoch_loss = losses.avg.item()

                            if epoch_loss < best_loss:
                                best_loss = epoch_loss
                                best_epoch = epoch

                                save_dir = os.path.join(SAVE_ROOT, FILE_NAME, "weights")
                                os.makedirs(save_dir, exist_ok=True)

                                # 1. ArcFace Classifier SAVE
                                arcface_classifier_save_path = os.path.join(save_dir, "best_arcface_classifier.pt")
                                torch.save({"weight": fc_metric.weight.detach().cpu()}, arcface_classifier_save_path)

                                # 2. Relative MLP SAVE
                                metric_learning_model_save_path = os.path.join(save_dir, "best_metric_learning_model.pth.tar")
                                torch.save(fc_metric.layers.state_dict(), metric_learning_model_save_path)

                                # 3. nn.Embedding SAVE
                                nn_embedding_save_path = os.path.join(save_dir, "best_nn_embedding.pt")
                                torch.save(fc_metric.embedding.state_dict(), nn_embedding_save_path)

                            #
                            scheduler.step()

                            loss_value = losses.val.item()
                            loss_average = losses.avg.item()
                            msg = (
                                f"Epoch: [{epoch}]  "
                                f"Time {batch_time.val:.3f}s   "
                                f"Speed {J_coord.size(0) / batch_time.val:.1f} samples/s\t"
                                f"Loss {loss_value:.7f} "
                                f"({loss_average:.7f})"
                            )
                            print(msg)

                            epoch_end_time = time.time()
                            # print(f"{epoch_end_time - epoch_start_time:.2f} sec")

                        print(
                            f"Training finished - "
                            f"Best Loss: {best_loss:.7f} "
                            f"(Epoch {best_epoch + 1})"
                        )

                        save_dir = os.path.join(SAVE_ROOT, FILE_NAME, "weights")
                        os.makedirs(save_dir, exist_ok=True)

                        # 1. ArcFace Classifier SAVE
                        arcface_classifier_save_path = os.path.join(save_dir, "final_arcface_classifier.pt")
                        torch.save({"weight": fc_metric.weight.detach().cpu()}, arcface_classifier_save_path)

                        # 2. Relative MLP SAVE
                        metric_learning_model_save_path = os.path.join(save_dir, "final_metric_learning_model.pth.tar")
                        torch.save(fc_metric.layers.state_dict(), metric_learning_model_save_path)

                        # 3. nn.Embedding SAVE
                        nn_embedding_save_path = os.path.join(save_dir, "final_nn_embedding.pt")
                        torch.save(fc_metric.embedding.state_dict(), nn_embedding_save_path)

                        # ---------------------------------------------------------
                        # Best model load for validation
                        # ---------------------------------------------------------
                        best_arcface_classifier_path = os.path.join(
                            save_dir,
                            "best_arcface_classifier.pt",
                        )
                        best_metric_learning_model_path = os.path.join(
                            save_dir,
                            "best_metric_learning_model.pth.tar",
                        )
                        best_nn_embedding_path = os.path.join(
                            save_dir,
                            "best_nn_embedding.pt",
                        )

                        # 1. ArcFace classifier weight
                        arcface_checkpoint = torch.load(
                            best_arcface_classifier_path,
                            map_location=device,
                        )

                        with torch.no_grad():
                            fc_metric.weight.copy_(
                                arcface_checkpoint["weight"].to(device)
                            )

                        # 2. Relative MLP
                        metric_learning_state = torch.load(
                            best_metric_learning_model_path,
                            map_location=device,
                        )
                        fc_metric.layers.load_state_dict(
                            metric_learning_state,
                            strict=True,
                        )

                        # 3. nn.Embedding
                        embedding_state = torch.load(
                            best_nn_embedding_path,
                            map_location=device,
                        )
                        fc_metric.embedding.load_state_dict(
                            embedding_state,
                            strict=True,
                        )

                        fc_metric.eval()

                        #
                        all_feats = []
                        all_labels = []
                        #
                        batch_time = AverageMeter()
                        data_time = AverageMeter()

                        with torch.inference_mode():
                            valid_start = time.time()
                            for i ,(J_coord) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                                # BS 계산
                                BS = J_coord.shape[0]

                                # 입력 데이터
                                J_coord = J_coord.to(device, non_blocking=True)
                                token_ids = base_token_ids.expand(BS, -1)

                                # Forward
                                _, embedding_vec = fc_metric(input=J_coord, token_ids=token_ids, use_arcface=False, m=M, s=S)

                                #
                                all_feats.append(embedding_vec.reshape(-1, embedding_vec.size(-1)).cpu())
                                all_labels.append(token_ids.reshape(-1).cpu())
                    #
                        all_feats = torch.cat(all_feats, dim=0)
                        all_labels = torch.cat(all_labels, dim=0)

                        score = plot_tsne_with_centroids(config=config, feats=all_feats, labels=all_labels, vocab=joint_vocab, save_root = SAVE_ROOT, file_name = FILE_NAME, visualization=config.VIS.PLOT_VISUALIZATION)
                        valid_end = time.time()
                        print(f'{valid_end - valid_start} sec')

                        experiment_metadata = {
                            "configuration": {
                                "feature_mode": FEATURE_MODE,
                                "embedding_mode": EMB_MODE,
                                "num_layers": int(NUM_LAYERS),
                                "input_dim": int(IN_DIM),
                                "output_dim": int(config.MODEL.OUT_CHANNELS),
                                "arcface_scale": float(S),
                                "arcface_margin": float(M),
                                "activation": str(config.TRAIN.ACT),
                            },

                            "training": {
                                "epochs": int(config.TRAIN.EPOCH),
                                "warmup_epochs": int(config.TRAIN.WARMUP_EPOCH),
                                "batch_size": int(config.TRAIN.BATCH_SIZE),
                                "learning_rate": float(config.TRAIN.LR),
                                "shuffle": bool(config.TRAIN.SHUFFLE),
                                "seed": int(config.SEED),
                                "optimizer": "Adam",
                                "scheduler": "LinearLR + CosineAnnealingLR",
                            },

                            "model_selection": {
                                "criterion": "minimum_epoch_average_training_loss",
                                "best_epoch": int(best_epoch + 1),
                                "best_loss": float(best_loss),
                                "validation_checkpoint": "best",
                            },

                            "validation": {
                                "num_pose_samples": int(num_pose_samples),
                                "batch_size": int(config.VALID.BATCH_SIZE),
                                "metric_method": str(config.VIS.PLOT_METRIC_METHOD),
                                "tsne_plot_dim": int(config.VIS.PLOT_DIM),
                                "tsne_perplexity": float(config.VIS.TSNE_PERPLEXITY),
                                "tsne_random_seed": int(
                                    config.VIS.TSNE_RANDOM_SEED
                                ),
                            },

                            "dataset": {
                                "train_path": TRAIN_DATA_PATH,
                                "valid_path": VALID_DATA_PATH,
                            },

                            "paths": {
                                "result_root": os.path.join(
                                    SAVE_ROOT,
                                    FILE_NAME,
                                ),
                                "weights_dir": save_dir,
                            },
                        }

                        metrics_dir = os.path.join(
                            SAVE_ROOT,
                            FILE_NAME,
                            "metrics",
                        )
                        os.makedirs(metrics_dir, exist_ok=True)

                        metadata_path = os.path.join(
                            metrics_dir,
                            "experiment_config.yaml",
                        )

                        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                            yaml.safe_dump(
                                experiment_metadata,
                                metadata_file,
                                allow_unicode=True,
                                sort_keys=False,
                            )

                        configuration_elapsed = (
                                time.time() - configuration_start_time
                        )

                        recent_configuration_times.append(
                            configuration_elapsed
                        )
                        completed_configurations += 1

                        average_configuration_time = (
                                sum(recent_configuration_times)
                                / len(recent_configuration_times)
                        )

                        remaining_configurations = (
                                total_configurations
                                - completed_configurations
                        )

                        estimated_remaining_seconds = (
                                average_configuration_time
                                * remaining_configurations
                        )

                        total_elapsed_seconds = (
                                time.time() - experiment_start_time
                        )

                        estimated_finish_time = (
                                datetime.now()
                                + timedelta(
                            seconds=estimated_remaining_seconds
                        )
                        )

                        print(
                            f"\nProgress: "
                            f"{completed_configurations}/"
                            f"{total_configurations}"
                        )

                        print(
                            f"Current configuration time: "
                            f"{timedelta(seconds=int(configuration_elapsed))}"
                        )

                        print(
                            f"Recent average configuration time: "
                            f"{timedelta(seconds=int(average_configuration_time))}"
                        )

                        print(
                            f"Total elapsed time: "
                            f"{timedelta(seconds=int(total_elapsed_seconds))}"
                        )

                        print(
                            f"Estimated remaining time: "
                            f"{timedelta(seconds=int(estimated_remaining_seconds))}"
                        )

                        print(
                            f"Estimated finish time: "
                            f"{estimated_finish_time:%Y-%m-%d %H:%M:%S}"
                        )
