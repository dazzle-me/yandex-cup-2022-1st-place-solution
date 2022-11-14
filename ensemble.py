import os
import os.path as osp

import sys
import argparse
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.cuda.amp import autocast

from einops import rearrange

from train_arcface import prepare_loaders
from utils import listdir
from utils.yc_utils import get_ranked_list, eval_submission, save_submission, get_ranked_list_exact

TOP_SIZE = 100
ANNOY_NUM_TREES = 512
USE_EXACT = True

@torch.inference_mode()
def get_embeddings(model, dataloader, device):
    embeds = dict()
    for data in tqdm(dataloader):
        image = data['image'].to(device)
        mask = data['mask'].to(device, dtype=torch.bool)
        with autocast(True):
            embeddings = model(image, mask)
            ## normalize here to avoid unnesesary allocation inside get_ranked_list_exact
            embeddings = F.normalize(embeddings, dim=1).detach().cpu().numpy()
        for trackid, embedding in zip(data['trackid'].cpu().numpy(), embeddings):
            embeds[trackid] = embedding
    return embeds

class EnsembleArcface(nn.Module):
    def __init__(self, model_list, mode):
        super().__init__()
        self.models = model_list
        self.mode = mode
    
    def forward_cat(self, x, mask):
        embeddings = []
        for model in self.models:
            embeddings.append(model.extract(x, mask))
        return F.normalize(torch.cat(embeddings, dim=1), dim=1)

    def forward_mean(self, x, mask):
        embeddings = []
        for model in self.models:
            embedding = model.extract(x, mask)
            embedding = rearrange(embedding, 'b f -> b f 1')
            embeddings.append(F.normalize(embedding, dim=1))
        embeddings = torch.cat(embeddings, dim=2)
        return F.normalize(torch.mean(embeddings, dim=2), dim=1)

    def forward(self, x, mask):
        if self.mode == 'cat':
            return self.forward_cat(x, mask)
        else:
            return self.forward_mean(x, mask)

def load_module(module, path):
    spec = importlib.util.spec_from_file_location(module, path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    return foo

pp_weights = np.array([25.0, 8.0, 5.0, 3.0, 2.0, 1.0], dtype=np.float32)
def get_pp_ranked_list(ranked_list, embeds, device='cuda:0'):
    global pp_weights
    print(f"Using neighbouring weights : {pp_weights}")
    # new_embeds
    new_embeds = dict()
    for k, v in ranked_list.items():
        new_embeds[k] = np.zeros_like(embeds[k])
        for idx in range(len(pp_weights)):
            if idx == 0:
                new_embeds[k] = embeds[k] * pp_weights[0]
            else:
                new_embeds[k] += embeds[v[idx - 1]] * pp_weights[idx]
        new_embeds[k] /= sum(pp_weights)
    ranked_list = get_ranked_list_exact(new_embeds, TOP_SIZE, device)
    return ranked_list

@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, df_val):
    model.eval()
    print("Bulding embeddings")
    embeds = get_embeddings(model, dataloader, device)
    if USE_EXACT:
        print("Using GPU cosine-sim")
        ranked_list = get_ranked_list_exact(embeds, TOP_SIZE, device)
    else:
        print("Using annoy to get ranked list")
        ranked_list = get_ranked_list(embeds, TOP_SIZE, annoy_num_trees=ANNOY_NUM_TREES)
    print("Calculating the metric")
    nDCG = eval_submission(ranked_list, df_val)
    print(f"No clustering: {nDCG:6f}")
    ranked_list = get_pp_ranked_list(ranked_list, embeds, device='cuda:0')
    nDCG = eval_submission(ranked_list, df_val)
    print(f'After clustering: {nDCG:.6f}')
    return nDCG 

@torch.inference_mode()
def test_one_epoch(model, dataloader, device, sub_path):
    model.eval()
    print("Bulding embeddings")
    embeds = get_embeddings(model, dataloader, device)
    del model
    if USE_EXACT:
        print("Using GPU cosine-sim")
        ranked_list = get_ranked_list_exact(embeds, TOP_SIZE, device)
    else:
        print("Using annoy to get ranked list")
        ranked_list = get_ranked_list(embeds, TOP_SIZE, annoy_num_trees=ANNOY_NUM_TREES) 

    ranked_list = get_pp_ranked_list(ranked_list, embeds, device='cuda:0')
    print("Saving the sub")
    save_submission(ranked_list, sub_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True, default='./data')
    parser.add_argument('--exp-root', type=str, default='./exps')
    parser.add_argument('--exp', nargs='*')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save-dir', type=str, default='./ensemble')
    parser.add_argument('--eval-fold', type=int, default=-1, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--mode', type=str, choices=['mean', 'cat'], default='cat')
    parser.add_argument('--save-val-embeddings', action="store_true", default=False)
    parser.add_argument('--save-test-embeddings', action="store_true", default=False)
    parser.add_argument('--num-checkpoints', type=int, default=1)
    args = parser.parse_args()
    print(f"Using mode : {args.mode}")
    models = []
    names = []
    for exp in args.exp:
        for fold in range(10):
            if fold == args.eval_fold or args.eval_fold == -1:
                exp_dir = osp.join(args.exp_root, exp, f"fold_{fold}")
                if not osp.isdir(exp_dir):
                    continue
                names.append("_".join(exp_dir.split('/')[-2:]))
                sys.path.insert(0, exp_dir)
                try:
                    weights = sorted(listdir(osp.join(exp_dir, 'weights')), key=lambda x : float(osp.basename(x)[4:12]))[-args.num_checkpoints:]
                    for weight_file in weights:
                        config = load_module('train_arcface', osp.join(exp_dir, 'train_arcface.py')).CONFIG
                        model = load_module('train_arcface', osp.join(exp_dir, 'train_arcface.py')).YCModel(config['embedding_size']).eval()
                        print("Loaded model successfully!")        
                        print(f"Weights : {weight_file}")
                        # status = mode
                        status = model.load_state_dict(torch.load(weight_file))
                        model = model.to(args.device)
                        models.append(model)
                        print(f"Status : {status}")
                except:
                    pass
    print(f"Using {len(models)} models")
    ens = EnsembleArcface(models, args.mode).eval()
    
    train_df = pd.read_csv(f'{args.data_root}/train_meta.tsv', sep='\t')
    test_df = pd.read_csv(f'{args.data_root}/test_meta.tsv', sep='\t')
    from sklearn.model_selection import GroupKFold
    skf = GroupKFold(n_splits=10)

    for fold, ( _, val_) in enumerate(skf.split(X=train_df, y=train_df.artistid, groups=train_df.artistid)):
        train_df.loc[val_ , "kfold"] = fold
    train_loader, valid_loader, test_loader, df_train, df_valid, df_test = prepare_loaders(train_df, test_df, fold=args.eval_fold if args.eval_fold != -1 else 0)
    if args.eval_fold != -1:
        nDCG = valid_one_epoch(ens, valid_loader, args.device, df_valid)
        print(f"Metric : {nDCG}")
    else:
        nDCG = -1
    if not osp.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    name = names if args.eval_fold != -1 else args.exp
    save_name = osp.join(args.save_dir, f"nDCG_{nDCG:.6f}_mode_{args.mode}|" + "|".join(name) + f'_pp_{list(pp_weights)}.csv')
    print(f"Saving test.csv... : {save_name}")
    test_one_epoch(ens, test_loader, args.device, save_name)

