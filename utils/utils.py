import os
import os.path as osp

def listdir(root, hard=True):
    if hard:
        return [osp.join(root, file) for file in os.listdir(root)]
    return os.listdir(root)

def batch_to_device(batch, cfg):
    for k, v in batch.items():
        if k in cfg.fields_to_move:
            batch[k] = v.to(cfg.device)
    return batch