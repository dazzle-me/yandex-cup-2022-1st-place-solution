from collections import defaultdict
import torch
import torch.nn.functional as F

import annoy
import numpy as np
from tqdm import tqdm

def vec_translate(a, d):    
    return np.vectorize(d.__getitem__)(a)

def save_submission(submission, submission_path):
    with open(submission_path, 'w') as f:
        for query_trackid, result in submission.items():
            f.write("{}\t{}\n".format(query_trackid, " ".join(map(str, result))))

@torch.inference_mode()
def get_ranked_list_exact(embeds, top_size, device='cuda:0'):
    num_chunks = 512
    index2track = list(embeds.keys())
    embeds = np.stack(list(embeds.values()), axis=0)
    embeds = torch.from_numpy(embeds).to(device)
    ranked_list = dict()
    track_id_index = 0
    for chunk in tqdm(embeds.chunk(num_chunks)):
        chunk = chunk.to(device)
        cos_sim_chunk_values = torch.mm(chunk, embeds.transpose(0, 1))
        
        cos_sim_chunk_values, cos_sim_chunk = cos_sim_chunk_values.sort(dim=1, descending=True)
        cos_sim_chunk = cos_sim_chunk[:, 1:top_size + 50].detach().cpu().numpy()
        cos_sim_chunk_values = cos_sim_chunk_values[:, 1:top_size+ 50]
        # print(cos_sim_chunk_values.shape, cos_sim_chunk.shape)
        for similarity_index, similarity in zip(cos_sim_chunk, cos_sim_chunk_values):
            similarity_index = similarity_index[similarity_index != track_id_index]
            current_trackid = index2track[track_id_index] 
            ranked_list[current_trackid] = vec_translate(similarity_index[:100], index2track)
            track_id_index += 1
    return ranked_list

def get_ranked_list(embeds, top_size, *, annoy_num_trees = 512):
    annoy_index = None
    annoy2id = []
    id2annoy = dict()
    for track_id, track_embed in embeds.items():
        id2annoy[track_id] = len(annoy2id)
        annoy2id.append(track_id)
        if annoy_index is None:
            annoy_index = annoy.AnnoyIndex(len(track_embed), 'angular')
        annoy_index.add_item(id2annoy[track_id], track_embed)
    annoy_index.build(annoy_num_trees)
    ranked_list = dict()
    for track_id in embeds.keys():
        candidates = annoy_index.get_nns_by_item(id2annoy[track_id], top_size+1)[1:] # exclude trackid itself
        candidates = list(filter(lambda x: x != id2annoy[track_id], candidates))
        ranked_list[track_id] = [annoy2id[candidate] for candidate in candidates]
    return ranked_list

def position_discounter(position):
    return 1.0 / np.log2(position+1)   

def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg

def compute_dcg(query_trackid, ranked_list, track2artist_map, top_size):
    query_artistid = track2artist_map[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list[:top_size]):
        # if result_trackid == query_trackid:
        #     print(result_trackid, query_trackid)
        assert result_trackid != query_trackid
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist_map[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg

def eval_submission(submission, gt_meta_info, top_size = 100):
    track2artist_map = gt_meta_info.set_index('trackid')['artistid'].to_dict()
    artist2tracks_map = gt_meta_info.groupby('artistid').agg(list)['trackid'].to_dict()
    ndcg_list = []
    for query_trackid in tqdm(submission.keys()):
        ranked_list = submission[query_trackid]
        query_artistid = track2artist_map[query_trackid]
        query_artist_tracks_count = len(artist2tracks_map[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count-1, top_size=top_size)
        dcg = compute_dcg(query_trackid, ranked_list, track2artist_map, top_size=top_size)
        try:
            ndcg_list.append(dcg/ideal_dcg)
        except ZeroDivisionError:
            continue
    return np.mean(ndcg_list)