# -*- coding: utf-8 -*-
'''
This scripts performs kNN search on inferenced image and text features (on single-GPU) and outputs text-to-image prediction file for evaluation.
'''

import argparse
import numpy
from tqdm import tqdm
import json

import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-feats', 
        type=str, 
        required=True,
        help="Specify the path of image features."
    )  
    parser.add_argument(
        '--text-feats', 
        type=str, 
        required=True,
        help="Specify the path of text features."
    )      
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=10,
        help="Specify the k value of top-k predictions."
    )   
    parser.add_argument(
        '--eval-batch-size', 
        type=int, 
        default=32768,
        help="Specify the image-side batch size when computing the inner products, default to 8192"
    )    
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help="Specify the output jsonl prediction filepath."
    )
    parser.add_argument(
        '--npy_path', 
        type=str, 
        required=True,
        help="Specify the input prediction matrix filepath."
    )         
    parser.add_argument(
        '--npy_save', 
        type=str, 
        required=True,
        help="Specify the output prediction matrix filepath."
    )                 
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print(args.npy_save)
    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    print("Begin to load image features...")
    image_ids = []
    image_feats = []
    text_ids = []
    text_feats = []
    with open(args.image_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            image_ids.append(obj['image_id'])
            image_feats.append(obj['feature'])
    image_feats_array = np.array(image_feats, dtype=np.float32)
    image_feats = torch.from_numpy(image_feats_array).cuda()
    print("Finished loading image features.")

    with open(args.text_feats, "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            text_ids.append(obj['text_id'])
            text_feats.append(obj['feature'])
    text_feats_array = np.array(text_feats, dtype=np.float32)
    text_feats = torch.from_numpy(text_feats_array).cuda()
    print("Finished loading text features.")

    print("Begin to compute top-{} predictions for texts...".format(args.top_k))

    image_norm = torch.norm(image_feats, p=2, dim=1, keepdim=True)
    text_norm = torch.norm(text_feats, p=2, dim=1, keepdim=True)

    image_feats = image_feats.div(image_norm.expand_as(image_feats))
    text_feats = text_feats.div(text_norm.expand_as(text_feats))

    qq_dist = (text_feats @ text_feats.t()).cpu()
    qg_dist = (text_feats @ image_feats.t()).cpu()
    gg_dist = (image_feats @ image_feats.t()).cpu()

    qq_dist = qq_dist.numpy()
    qg_dist = qg_dist.numpy()
    gg_dist = gg_dist.numpy()

    #final_dist = re_ranking(qg_dist, qq_dist, gg_dist, k1=10, k2=3, lambda_value=0.3)
    essemble_dist = np.load(f'{args.npy_path}/matrix-1.npy')
    essemble_dist2 = np.load(f'{args.npy_path}/matrix-2.npy')
    essemble_dist3 = np.load(f'{args.npy_path}/matrix-3.npy')
    essemble_dist4 = np.load(f'{args.npy_path}/matrix-4.npy')
    essemble_dist5 = np.load(f'{args.npy_path}/matrix-5.npy')
    essemble_dist6 = np.load(f'{args.npy_path}/matrix-6.npy')
    essemble_dist7 = np.load(f'{args.npy_path}/matrix-7.npy')
    essemble_dist8 = np.load(f'{args.npy_path}/matrix-8.npy')
    final_dist = (0.5 * essemble_dist + 0.5 * essemble_dist2 + 0.5 * essemble_dist3 + 0.5 * essemble_dist4 + 0.5 * essemble_dist5 + 1.25 * essemble_dist6 + 0.75 * essemble_dist7 + 0.7 * essemble_dist8)/8
    #final_dist = qg_dist
    np.save(args.npy_save, qg_dist)

    image_ids = np.array(image_ids)
    text_ids = np.array(text_ids)

    with open(args.output, "w") as fout:
        for idx, text_id in enumerate(text_ids):
            dist = final_dist[idx]
            argids = np.argsort(-dist)[0:10]
            image_preds = image_ids[argids].tolist()
            fout.write("{}\n".format(json.dumps({"query_id": text_id.item(), "item_ids": image_preds})))
    
    print("Top-{} predictions are saved in {}".format(args.top_k, args.output))
    print("Done!")
