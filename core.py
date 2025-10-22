from email.policy import default

import torch

import yaml
import time
import random
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.nn.modules import loss

from config.config import config
from easydict import EasyDict as edict

from utils.function import plot_tsne_with_centroids
from utils.AverageMeter import AverageMeter

from model.CosineFace import CosFace
from model.ArcFace import ArcFace
from loader.Coord_Dataset import Coord_Dataset

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


    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    fix_seed(config.SEED)

    gen_config('/home/jysuh/PycharmProjects/coord_embedding/coord_embed.yaml')

    NUM_JOINTS = config.DATASET.NUM_JOINTS
    NUM_TOKEN = config.DATASET.NUM_TOKEN

    device = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")

    if config.TRAIN.LOSS == 'CosFace':
        fc_metric = CosFace(in_features=4, out_features=512, num_class=NUM_JOINTS, use_embedding=config.TRAIN.USE_EMB, activation=config.TRAIN.ACT, s=10.0, m=0.20, device=device).to(device)

    elif config.TRAIN.LOSS == 'ArcFace':
        fc_metric = ArcFace(num_layer=config.MODEL.NUM_LAYER, in_features=config.MODEL.IN_CHANNELS, out_features=config.MODEL.OUT_CHANNELS, num_class=NUM_JOINTS+NUM_TOKEN, use_embedding=config.TRAIN.USE_EMB, activation=config.TRAIN.ACT, s=10.0, m=0.10,device=device).to(device)


    if config.PRETRAINED:
        state_dict = torch.load(config.PRETRAINED_PATH, map_location=device)
        fc_metric.load_state_dict(state_dict)
        print('load weight...' + config.PRETRAINED_PATH)
        print(fc_metric)

    # if config.PRETRAINED_EMB:
    #     emb_weight = torch.load(config.PRETRAINED_EMB_PATH)
    #     statedict = fc_metric.state_dict()
    #     statedict['embedding.weight'] = emb_weight
    #     fc_metric.load_state_dict(statedict, strict=True)
    #     print('load embedding weight...' + config.PRETRAINED_EMB_PATH)
    #     # embedding = embedding.from_pretrained(emb_weight)


    train_dataset = Coord_Dataset(config=config, data_path=config.DATASET.TRAIN_DATA_PATH)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    valid_dataset = Coord_Dataset(config=config, data_path=config.DATASET.VALID_DATA_PATH)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        collate_fn=valid_dataset.collate_fn
    )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(fc_metric.parameters(), lr=config.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)
    #
    token_idx = [i for i in range(NUM_TOKEN)]
    token_idx = torch.LongTensor(token_idx)
    #
    if config.MODE == 'TRAIN':
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        #
        for epoch in range(config.TRAIN.EPOCH):
            if config.TRAIN.WARMUP:
                m = (epoch / config.TRAIN.WARMUP_EPOCH) * 0.1 + 0.1
                s = (epoch / config.TRAIN.WARMUP_EPOCH) * 10 + 10
            batch_time.reset()
            data_time.reset()
            losses.reset()
            #
            end = time.time()
            #
            for i ,(J_coord, J_tokens, WRKOUT, FRAME, VIEW, VIDEO) in enumerate(train_loader):
                if len(VIDEO) == 0:
                    continue
                MINI_BATCH = WRKOUT.shape[0]
                #
                loss = 0
                #
                dummy = torch.zeros([MINI_BATCH, NUM_TOKEN, 4]).to(device)
                J_coord = torch.cat([dummy, J_coord.to(device)], dim=1)
                #
                J_tokens = torch.cat((token_idx, J_tokens[0]))
                J_tokens = J_tokens.unsqueeze(0).repeat(MINI_BATCH, 1).to(device)

                optimizer.zero_grad()

                for idx in range(NUM_TOKEN + NUM_JOINTS):
                    if config.TRAIN.WARMUP:
                        logits, _ = fc_metric(input=J_coord[:, idx, :], J_tokens=J_tokens[:, idx], mode='training',
                                              m=m, s=s)
                    else:
                        logits, _ = fc_metric(input=J_coord[:, idx, :], J_tokens=J_tokens[:, idx], mode='training')
                    #
                    label = J_tokens[:, idx]
                    loss += criterion(logits, label)
                    # print(criterion(logits, label))
                #
                loss /= NUM_JOINTS
                loss.backward()
                losses.update(loss.item(), J_coord.size(0))
                #
                batch_time.update(time.time() - end)
                end = time.time()
                #
                if i % config.PRINT_FREQ == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss {loss.val:.7f} ({loss.avg:.7f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        speed=J_coord.size(0) / batch_time.val,
                        data_time=data_time, loss=losses)
                    # logger.info(msg)
                    print(msg)
                optimizer.step()
            scheduler.step()
        save_dir = f"/home/jysuh/PycharmProjects/coord_embedding/checkpoint/{config.FILE_NAME}"
        os.makedirs(save_dir, exist_ok=True)

        metric_learning_model_save_path = os.path.join(save_dir, "metric_learning_model.pth.tar")
        torch.save(fc_metric.state_dict(), metric_learning_model_save_path)
        #
        nn_embedding_save_path = os.path.join(save_dir, "nn_embedding.pt")
        torch.save(fc_metric.embedding.state_dict(), nn_embedding_save_path)



    elif config.MODE == 'TEST':
        fc_metric.eval()
        #
        all_feats = []
        all_labels = []
        #
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # To collect data for model test
        TOTAL_ITERS = len(valid_loader)
        random_indices = set(random.sample(range(TOTAL_ITERS), config.VALID.NUM_SAMPLE))
        #
        if config.GEN_BERT_DATASET:
            from collections import defaultdict
            video_counter = defaultdict(int)
            embedding_vectors = {}

        c_WRKOUT = 0
        with torch.no_grad():
            # J_coord = [x,y,j_idx,wrkout_idx]
            for i ,(J_coord, J_tokens, WRKOUT, FRAME, VIEW, VIDEO) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                if len(VIDEO) == 0:
                    continue

                if config.GEN_BERT_DATASET:
                    a_frame = []
                    WRKOUT_TKN = int(WRKOUT[0])
                    VIEW_TKN = int(VIEW[0])
                    FRAME_TKN = int(FRAME[0])
                    #
                    VIDEO_IDX = int(VIDEO[0])
                    #
                    embedding_vectors.setdefault(WRKOUT_TKN, {})
                    embedding_vectors[WRKOUT_TKN].setdefault('Video', {})
                    #
                    video_counter[WRKOUT_TKN] += 1
                    #
                    embedding_vectors[WRKOUT_TKN]['Video'].setdefault(VIDEO_IDX, {})
                    embedding_vectors[WRKOUT_TKN]['Video'][VIDEO_IDX].setdefault(VIEW_TKN, {})
                    embedding_vectors[WRKOUT_TKN]['Video'][VIDEO_IDX][VIEW_TKN].setdefault(FRAME_TKN, [])
                    if c_WRKOUT != WRKOUT_TKN:
                        one_hot = torch.zeros(len(valid_dataset.vocab.keys()))      # the number of wrkout = 26
                        one_hot[WRKOUT_TKN-20] = 1
                        embedding_vectors[WRKOUT_TKN].setdefault('Label', one_hot)
                    #
                #
                MINI_BATCH = WRKOUT.shape[0]
                #
                dummy = torch.zeros([MINI_BATCH, NUM_TOKEN, 4]).to(device)
                J_coord = torch.cat([dummy, J_coord.to(device)], dim=1)
                #
                J_tokens = torch.cat((token_idx, J_tokens[0]))
                J_tokens = J_tokens.unsqueeze(0).repeat(MINI_BATCH, 1).to(device)

                for idx in range(NUM_TOKEN + NUM_JOINTS):
                    _, embedding_vec = fc_metric(input=J_coord[:, idx, :], J_tokens=J_tokens[:, idx], mode='validation')
                    if i in random_indices:
                        all_feats.append(embedding_vec.detach().cpu())          # List of [BS, 512]
                        all_labels.append(J_tokens[:, idx].detach().cpu())      # List of [BS]
                    #
                    if config.GEN_BERT_DATASET:
                        a_frame.append(embedding_vec.detach().cpu())
                #
                if config.GEN_BERT_DATASET:
                    embedding_vectors[WRKOUT_TKN]['Video'][VIDEO_IDX][VIEW_TKN][FRAME_TKN].append(a_frame)
                    c_WRKOUT = WRKOUT_TKN
            #
        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        score = plot_tsne_with_centroids(config=config, feats=all_feats, labels=all_labels, vocab=valid_dataset.vocab, visualization=config.VIS.PLOT_VISUALIZATION)
        print(score)

        if config.GEN_BERT_DATASET:
            diff = False
            renew = {}

            for wrkout in embedding_vectors.keys():
                renew.setdefault(wrkout, {})
                renew[wrkout].setdefault('Video', {})
                for video in embedding_vectors[wrkout]['Video'].keys():
                    renew[wrkout]['Video'].setdefault(video, {})
                    for view in embedding_vectors[wrkout]['Video'][video].keys():
                        renew[wrkout]['Video'][video].setdefault(view, {})
                        for key, value in sorted(embedding_vectors[wrkout]['Video'][video][view].items(), key=lambda x: x[0]):
                            renew[wrkout]['Video'][video][view][key] = value

            for wrkout in embedding_vectors.keys():
                for video in embedding_vectors[wrkout]['Video'].keys():
                    for view in embedding_vectors[wrkout]['Video'][video].keys():
                        for frame in embedding_vectors[wrkout]['Video'][video][view].keys():
                            for joint in range(len(embedding_vectors[wrkout]['Video'][video][view][frame])):
                                if not renew[wrkout]['Video'][video][view][frame][joint] == embedding_vectors[wrkout]['Video'][video][view][frame][joint]:
                                    diff = True

            if not diff:
                torch.save(renew, f'/home/jysuh/PycharmProjects/coord_embedding/bert_dataset/{config.FILE_NAME}.pth.tar')
                print('renew == embedding vectors !! SAVE DONE')