import os
import gc
import math
import copy
import time
import shutil

# For data manipulation
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
# Pytorch Imports
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# Utils
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


from einops import rearrange

NUM_EPOCHS = 21
USE_VAL = False
CONFIG = {"seed": 2022,
          "epochs": NUM_EPOCHS,
          "img_size": 448,
          "model_name": "tf_efficientnet_b0_ns",
        #   "num_classes": 16622,
          "num_classes" : 16622,
          "embedding_size": 512,
          "train_batch_size": 2048,
          "valid_batch_size": 128,
          "eval_every" : 3,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-7,
          "T_max" : 15,
        #   "T_max": NUM_EPOCHS * 163 if USE_VAL else NUM_EPOCHS * 130 ,
          "weight_decay": 0,
          "n_fold": 10,
          "n_accumulate": 1,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          # ArcFace Hyperparameters
          "optim" : "Adam",
          "learning_rate": 3e-4,#0.01,#3e-4,#0.01,
          "momentum" : 0.9,
          "nesterov" : False,
          "weight_decay" : 0,#1e-5,
          "s": 30.0, 
          "min_margin" : 0.5,
          "max_margin" : 0.5,
          "m": 0.5,
          "ls_eps": 0.0,
          "easy_margin": False,
          "exp_dir" : "exps/exp60",
          "use_val" : USE_VAL,
          "tb_dir" : ".tb_logs",
          "use_amp" : True,
          "keep_left_bound" : 81,
          "keep_right_bound" : 82,
          "sub_dir" : 'submissions',
          "weight_dir" : 'weights',
          'num_parallel_embedders' : 1,
          'num_consecutive_embedders' : 2,
          'num_subcenters' : 3,
          'annoy_num_trees' : 32,
          'use_exact_search' : True,
          'num_lstm_layers' : 2,
          'margin_slope' : 0.0,
          'margin_bias' : 0.5,
          'lowest_valid_thresh' : int(250 / 15 * 8),
          'eval_period' : 60 / 15 * 4,
}

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])

import os
import os.path as osp
ROOT_DIR = '../data'
TRAIN_DIR = osp.join(ROOT_DIR, 'train_features')
TEST_DIR = osp.join(ROOT_DIR, 'test_features')
if __name__ == '__main__':
    train_df = pd.read_csv(f"{ROOT_DIR}/train_meta.tsv", sep='\t')
    test_df = pd.read_csv(f"{ROOT_DIR}/test_meta.tsv", sep='\t')

    from sklearn.model_selection import GroupKFold
    skf = GroupKFold(n_splits=CONFIG['n_fold'])

    for fold, ( _, val_) in enumerate(skf.split(X=train_df, y=train_df.artistid, groups=train_df.artistid)):
        train_df.loc[val_ , "kfold"] = fold
    for fold in range(CONFIG['n_fold']):
        print(f"fold : {fold} : {train_df[train_df.kfold != fold].artistid.nunique()}")

import os
import os.path as osp

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, mode='train'):
        self.image_dir = image_dir
        self.df = df
        self.mode = mode
        if mode == 'train':
            self.le = LabelEncoder()
            self.df.artistid = self.le.fit_transform(self.df.artistid)
            frequency = np.sqrt(1 / np.sqrt(df['artistid'].value_counts().sort_index().values))
            ## from landmakrs solution
            self.margins = (frequency - frequency.min()) / (frequency.max() - frequency.min()) * CONFIG['margin_slope'] + CONFIG['margin_bias']
            # self.df.margin = self.df.artistid.apply(lambda x : self.margins[x])
        if mode in ['train', 'val']:
            print(f"num_classes : {self.df.artistid.nunique()}")
            print(f"class distribution : \n{self.df.artistid.value_counts().value_counts()}")

    def __getitem__(self, index):
        trackid = self.df['trackid'][index]
        # margin = self.df['margin'][index]
        path = self.df['archive_features_path'][index]
        mag_spec = np.load(osp.join(self.image_dir, path))
        h, w = mag_spec.shape
        ## zero-padding to (512, 81)
        if mag_spec.shape != (512, 81):
            mag_spec = np.hstack([mag_spec, np.zeros((512, 81 - w))])
        mag_spec = torch.Tensor(mag_spec)
        ## mask all zero-padding tokens in attention pooling
        mask = np.ones((81, ))
        mask[:w] = 0
        if self.mode in ['train', 'val']:
            label = self.df['artistid'][index]
            return {
                'trackid' : trackid,
                'image' : mag_spec,
                # 'margin' : margin,
                'mask' : mask,
                'label' : label # for cross-entropy-loss use "label" torch.Tensor([label])
            }
        return {
            'trackid' : trackid,
            'mask' : mask,
            'image' : mag_spec
        }


    def __len__(self): 
        return len(self.df)

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            margins: dynamic margins
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, margins, s=30.0, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.ls_eps = ls_eps  # label smoothing
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.margins = margins
    def forward(self, logits, labels):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        ms = []
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_features).float()
        # logits = logits.float()
        cosine = F.linear(F.normalize(logits), F.normalize(self.weight))
        
        # cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
        phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        return output

class AttentionPooling(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embedding_size, embedding_size), \
            nn.LayerNorm(embedding_size),
            nn.GELU(), 
            nn.Linear(embedding_size, 1)
        )
    def forward(self, x, mask=None):
        attn_logits = self.attn(x)
        if mask is not None:
            attn_logits[mask] = -float('inf')
        attn_weights = torch.softmax(attn_logits, dim=1)
        x = x * attn_weights
        # x = self.dropout(x)
        x = x.sum(dim=1)
        return x

class MHSA_branch(nn.Module):
    def __init__(
        self, embedding_size, n_heads=4, dropout=0.0, 
        num_layers=1, dim_feedforward=2048,
    ):
        super().__init__()
        self.norm_layers = nn.ParameterList([
            nn.LayerNorm(embedding_size) for _ in range(num_layers)
        ])
        self.encoder_layers = nn.ParameterList([
            nn.TransformerEncoderLayer(
                embedding_size, 
                nhead=n_heads, 
                dim_feedforward=dim_feedforward,
                activation="gelu",
                batch_first=True, 
                dropout=dropout,
                norm_first=False) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer, norm in zip(self.encoder_layers, self.norm_layers):
            x = norm(x)
            x = layer(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, emb_dim, mult=4, p=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * mult),
            nn.Dropout(p),
            nn.GELU(),
            nn.Linear(emb_dim * mult, emb_dim)
        )
    def forward(self, x):
        return self.fc(x)
        
class YCModel(nn.Module):
    def __init__(self, embedding_size, output_dim=512, input_size=512, margins=None):
        super().__init__()
        self.embedding_size = embedding_size

        num_branches = 2
        self.proj = FeedForward(input_size, mult=4)
        self.mhsa_branches = nn.ParameterList([MHSA_branch(embedding_size) for _ in range(num_branches)]) 
        self.pooling = AttentionPooling(embedding_size)
        self.bn = nn.BatchNorm1d(output_dim)
        self.arcface_head = ArcMarginProduct(
            output_dim, 
            CONFIG["num_classes"],
            margins,
            s=CONFIG["s"], 
            ls_eps=CONFIG["ls_eps"],
        )
    def get_margin(self):
        return self.arcface_head.m

    def forward(self, x, labels, mask):
        x = self.extract(x, mask)
        return self.arcface_head(x, labels)

    def extract(self, x, mask):
        x = rearrange(x, 'b f t -> b t f')
        x = self.proj(x)
        prediction = None
        for branch in self.mhsa_branches:
            if prediction is None:
                prediction = branch(x)
            else:
                prediction += branch(x)
        x = prediction
        x = self.pooling(x, mask)
        x = self.bn(x)
        return x

TIME_SINCE_LAST_VALID = 0
TOTAL_VALID = 0
def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def train_one_epoch(model, optimizer, scheduler, scaler, dataloader, device, epoch, valid_loader, df_val):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    global TIME_SINCE_LAST_VALID
    global TOTAL_VALID
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    # model.update_margin(epoch)
    # writer.add_scalar("arcface_margin", model.get_margin(), epoch)
    for step, data in bar:
        images = data['image']
        labels = data['label'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.bool)
        keep_tokens = np.random.randint(CONFIG['keep_left_bound'], CONFIG['keep_right_bound'])
        
        # randomly drop some tokens
        indices = np.random.choice(range(81), size=keep_tokens)
        # if np.random.uniform() <= 0.01:
        #     print(f"keeping : {keep_tokens} tokens")
        mask = mask[:, indices]
        images = images[:, :, indices].to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with autocast(enabled=CONFIG['use_amp']):
            outputs = model(images, labels, mask)
        # print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)
        loss = loss / CONFIG['n_accumulate']
            
        scaler.scale(loss).backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            scaler.step(optimizer)

            # zero the parameter gradients
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        TIME_SINCE_LAST_VALID += 1
        TOTAL_VALID += 1
        lr = optimizer.param_groups[0]['lr']
        if TOTAL_VALID == 1 or (lr < 1e-6 and TIME_SINCE_LAST_VALID >= CONFIG['eval_period'] and TOTAL_VALID >= CONFIG['lowest_valid_thresh']):
            print(f"Steps since last eval : {TIME_SINCE_LAST_VALID}")
            nDCG = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
                                            df_val=df_val)
            # if nDCG >= best_epoch_loss:
            # print(f"Validation Loss Improved ({best_epoch_loss} ---> {nDCG})")
            # best_epoch_loss = nDCG
            # run.summary["Best Loss"] = best_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = osp.join(CONFIG['exp_dir'], CONFIG['weight_dir'], "Loss{:.6f}_epoch{:.0f}.bin").format(nDCG, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved {PATH}")
            TIME_SINCE_LAST_VALID = 0
            # if nDCG != 0:
            print(f"Valid nDCG: {nDCG}")
            writer.add_scalar(f"nDCG_exact/{CONFIG['val_fold']}", nDCG, epoch)
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=lr)
    gc.collect()
    
    return epoch_loss

@torch.inference_mode()
def get_embeddings(model, dataloader, device):
    embeds = dict()
    for data in tqdm(dataloader):
        image = data['image'].to(device)
        mask = data['mask'].to(device, dtype=torch.bool)
        with autocast(enabled=CONFIG['use_amp']):
            embeddings = model.extract(image, mask)
            embeddings = F.normalize(embeddings, dim=1).detach().cpu().numpy()
        for trackid, embedding in zip(data['trackid'].cpu().numpy(), embeddings):
            embeds[trackid] = embedding
    return embeds

TOP_SIZE = 100
from utils.yc_utils import get_ranked_list, eval_submission, save_submission, get_ranked_list_exact
@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, df_val):
    model.eval()
    print("Bulding embeddings")
    embeds = get_embeddings(model, dataloader, device)
    # print("Using annoy to get ranked list")
    if CONFIG['use_exact_search']:
        print("Using GPU cosine-sim")
        ranked_list = get_ranked_list_exact(embeds, TOP_SIZE, device)
    else:
        print("Using annoy to get ranked list")
        ranked_list = get_ranked_list(embeds, TOP_SIZE, annoy_num_trees=CONFIG['ANNOY_NUM_TREES'])
    print("Calculating the metric")
    nDCG = eval_submission(ranked_list, df_val)
    model.train()
    return nDCG 

@torch.inference_mode()
def test_one_epoch(model, dataloader, device, sub_path):
    model.eval()
    print("Bulding embeddings")
    embeds = get_embeddings(model, dataloader, device)
    if CONFIG['use_exact_search']:
        print("Using GPU cosine-sim")
        ranked_list = get_ranked_list_exact(embeds, TOP_SIZE, device)
    else:
        print("Using annoy to get ranked list")
        ranked_list = get_ranked_list(embeds, TOP_SIZE, annoy_num_trees=CONFIG['ANNOY_NUM_TREES'])
    print("Saving the sub")
    save_submission(ranked_list, sub_path)
     

def run_training(model, optimizer, scheduler, scaler, device, num_epochs, train_loader, valid_loader, test_loader, df_val):
            
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = -np.inf
    history = defaultdict(list)
    
    for epoch in range(num_epochs):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, scaler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch, valid_loader=valid_loader, df_val=df_val)
        # if epoch % CONFIG["eval_every"] == 0 or epoch >= 20:
        #     nDCG = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
        #                                     df_val=df_val)
        #     if nDCG >= best_epoch_loss:
        #         print(f"Validation Loss Improved ({best_epoch_loss} ---> {nDCG})")
        #         best_epoch_loss = nDCG
        #         # run.summary["Best Loss"] = best_epoch_loss
        #         best_model_wts = copy.deepcopy(model.state_dict())
        #         PATH = osp.join(CONFIG['exp_dir'], CONFIG['weight_dir'], "Loss{:.6f}_epoch{:.0f}.bin").format(best_epoch_loss, epoch)
        #         torch.save(model.state_dict(), PATH)
        #         # Save a model file from the current directory
        #         print(f"Model Saved {PATH}")
            
        #     # sub.to_csv(, index=False)
        # else:
        #     nDCG = 0
        history['Train Loss'].append(train_epoch_loss)
        # history['Valid Loss'].append(nDCG)
        writer.add_scalar(f"arcface_s{CONFIG['s']}_m{CONFIG['m']}_loss/{CONFIG['val_fold']}", train_epoch_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        # Log the metrics
        print(f"Train Loss: {train_epoch_loss}")
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler

DEBUG = False
def prepare_loaders(df, df_test, fold):
    if not CONFIG['use_val']:
        df_train = df[df.kfold != fold].reset_index(drop=True)
    else:
        df_train = df
    # df_
    if DEBUG:
        df_train = df_train.sample(1000).reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = CustomDataset(df_train, TRAIN_DIR, mode='train')
    valid_dataset = CustomDataset(df_valid, TRAIN_DIR, mode='val')
    test_dataset = CustomDataset(df_test, TEST_DIR, mode='test')

    NUM_WORKERS = 16
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=NUM_WORKERS, shuffle=True, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=NUM_WORKERS, shuffle=False, pin_memory=True,
                              persistent_workers=True)
    test_loader =  DataLoader(test_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)
    return train_loader, valid_loader, test_loader, df_train, df_valid, df_test


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--seed', type=int, default=2022)
    
    # set_seed(CONFIG['seed'])

    args = parser.parse_args()

    if args.exp != '':
        CONFIG['exp_dir'] = args.exp    
    set_seed(args.seed)
    CONFIG['exp_dir'] = osp.join(CONFIG['exp_dir'], f'fold_{args.fold}')
    CONFIG['val_fold'] = args.fold
    train_df = pd.read_csv(f"{ROOT_DIR}/train_meta.tsv", sep='\t')
    test_df = pd.read_csv(f"{ROOT_DIR}/test_meta.tsv", sep='\t')

    from sklearn.model_selection import GroupKFold
    skf = GroupKFold(n_splits=CONFIG['n_fold'])

    for fold, ( _, val_) in enumerate(skf.split(X=train_df, y=train_df.artistid, groups=train_df.artistid)):
        train_df.loc[val_ , "kfold"] = fold
    for fold in range(CONFIG['n_fold']):
        print(f"fold : {fold} : {train_df[train_df.kfold != fold].artistid.nunique()}")
    train_df.to_csv(f'{ROOT_DIR}/train_folds.csv', index=False)
    train_loader, valid_loader, test_loader, df_train, df_valid, df_test = prepare_loaders(train_df, test_df, fold=args.fold)

    model = YCModel(CONFIG['embedding_size'], input_size=512, margins=train_loader.dataset.margins)
    print(f"Using dynamic margin : {train_loader.dataset.margins}")
    model.to(CONFIG['device']);
    print(model)
    if args.weights != '':
        print(f"Loading weights {args.weights}...")
        pretrained_state_dict = torch.load(args.weights)
        try:
            # pretrained_state_dict.pop('arcface_head')
            pretrained_state_dict.pop('proj.weight')
            pretrained_state_dict.pop('proj.bias')
        except:
            pass
        status = model.load_state_dict(pretrained_state_dict, strict=False)
        print(status)
    my_list = ['arcface_head.weight']
    params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))

    params = [p[1] for p in params]
    base_params = [p[1] for p in base_params]
    print(params)
    if CONFIG['optim'] in ['Adam', 'AdamW']:
        optimizer = optim.AdamW(
            [
                {"params" : params, "lr" : CONFIG['learning_rate'] * 10},
                {"params" : base_params, "lr" : CONFIG['learning_rate']}

            ], lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay']
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], 
                            weight_decay=CONFIG['weight_decay'], momentum=CONFIG['momentum'],
                            nesterov=CONFIG['nesterov'])

    scheduler = fetch_scheduler(optimizer)
    scaler = GradScaler()

    if not osp.isdir(CONFIG['exp_dir']):
        os.makedirs(CONFIG['exp_dir'])
    if not osp.isdir(osp.join(CONFIG['exp_dir'], CONFIG['weight_dir'])):
        os.makedirs(osp.join(CONFIG['exp_dir'], CONFIG['weight_dir']))
    if not osp.isdir(osp.join(CONFIG['exp_dir'], CONFIG['sub_dir'])):
        os.makedirs(osp.join(CONFIG['exp_dir'], CONFIG['sub_dir']))
    shutil.copyfile('train_arcface.py', osp.join(CONFIG['exp_dir'], 'train_arcface.py'))
    global writer
    writer = SummaryWriter(osp.join(CONFIG['tb_dir'], CONFIG['exp_dir']))
    print(CONFIG)
    model, history = run_training(model, optimizer, scheduler, scaler,
                                device=CONFIG['device'],
                                num_epochs=CONFIG['epochs'], 
                                train_loader=train_loader,
                                valid_loader=valid_loader, 
                                test_loader=test_loader,
                                df_val=df_valid)


