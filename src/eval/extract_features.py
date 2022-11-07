# -*- coding: utf-8 -*-
'''
This script extracts image and text features for evaluation. (with single-GPU)
'''

import os
import re
import time
import argparse
import logging
import json
import apex

import torch
import torch.nn as nn

from math import ceil
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from zhon.hanzi import punctuation
from torch.cuda.amp import GradScaler, autocast
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from clip.model import convert_weights, CLIP
from training.scheduler import cosine_lr
from training.main import convert_models_to_fp32
from training.train import get_loss
from training.online_learning import OnlineLearner
from eval.data import get_eval_img_dataset, get_eval_txt_dataset

logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--extract-image-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract image features."
    )
    parser.add_argument(
        '--extract-text-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract text features."
    )
    parser.add_argument(
        '--image-data', 
        type=str, 
        default="../Multimodal_Retrieval/lmdb/test/imgs", 
        help="If --extract-image-feats is True, specify the path of the LMDB directory storing input image base64 strings."
    )
    parser.add_argument(
        '--text-data', 
        type=str, 
        default="../Multimodal_Retrieval/test_texts.jsonl", 
        help="If --extract-text-feats is True, specify the path of input text Jsonl file."
    )
    parser.add_argument(
        '--image-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-image-feats is True, specify the path of output image features."
    )    
    parser.add_argument(
        '--text-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-image-feats is True, specify the path of output text features."
    )
    parser.add_argument(
        "--img-batch-size", type=int, default=64, help="Image batch size."
    )
    parser.add_argument(
        "--text-batch-size", type=int, default=64, help="Text batch size."
    )
    parser.add_argument(
        "--context-length", type=int, default=64, help="The maximum length of input text (include [CLS] & [SEP] tokens)."
    )
    parser.add_argument("--lr", type=float, default=0.75e-5, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-06, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--warmup", type=int, default=100, help="Number of steps to warmup for.")
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14"],
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--online-learning",
        default=False,
        action="store_true",
        help="If True, online learning in the test stage.",
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=None, 
        help="Number of steps to train for (in higher priority to --max_epochs)."
    )
    parser.add_argument(
        "--max-epochs", 
        type=int, 
        default=25, 
        help="Number of full epochs to train for (only works if --max_steps is None)."
    )
    parser.add_argument(
        "--rank", 
        type=int, 
        default=0, 
        help="The No. of device used."
    )
    parser.add_argument(
        "--online-cache",
        action="store_true",
        default=False,
        help="whether to use the cached downloaded images",
    )
    parser.add_argument(
        "--aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank."
    )
    parser.add_argument(
        "--report-training-batch-acc", default=True, action="store_true", help="Whether to report training batch accuracy."
    )
    args = parser.parse_args()
    args.local_device_rank = max(args.local_rank, 0)

    return args    

def _convert_to_rgb(image):
    return image.convert('RGB')

def _build_transform(resolution):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    return Compose([
            Resize((resolution, resolution), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])

if __name__ == "__main__":
    args = parse_args()

    assert args.extract_image_feats or args.extract_text_feats, "--extract-image-feats and --extract-text-feats cannot both be False!"

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")
    
    args.gpu = 0
    torch.cuda.set_device(args.gpu)

    # Initialize the model.
    vision_model_config_file = Path(__file__).parent / f"../training/model_configs/{args.vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)
    
    text_model_config_file = Path(__file__).parent / f"../training/model_configs/{args.text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)
    
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        for k, v in json.load(ft).items():
            model_info[k] = v

    model = CLIP(**model_info)
    convert_weights(model)    

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)
    model.cuda(args.gpu)
    if args.precision == "fp16":
        convert_weights(model)

    # Get data.
    if args.extract_image_feats:
        print("Preparing image inference dataset.")
        img_data = get_eval_img_dataset(args)
    if args.extract_text_feats:
        print("Preparing text inference dataset.")
        text_data = get_eval_txt_dataset(args, max_txt_length=args.context_length)
    
    # Resume from a checkpoint.
    print("Begin to load model checkpoint from {}.".format(args.resume))
    assert os.path.exists(args.resume), "The checkpoint file {} not exists!".format(args.resume)
    # Map model to be loaded to specified single gpu.
    loc = "cuda:{}".format(args.gpu)
    checkpoint = torch.load(args.resume, map_location='cpu')
    start_epoch = checkpoint["epoch"]
    start_steps = checkpoint['step']
    name = checkpoint["name"]
    optimizer_state = checkpoint['optimizer']
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    print(
        f"==> checkpoint '{args.resume} has been loaded!')"
    )


    # Initialize optimizer and lr scheduler
    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if not args.online_learning:
        optimizer = None
        scheduler = None
    else:
        decay_step = 20000
        optimizer = apex.optimizers.FusedLAMB(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": 0.01},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

        print(text_data.dataloader.num_batches)
        num_batches = text_data.dataloader.num_batches
        if args.max_steps is not None:
            args.max_epochs = ceil(args.max_steps / num_batches)
        else:
            assert args.max_epochs is not None and args.max_epochs > 0
            args.max_steps = num_batches * args.max_epochs
        total_steps = args.max_steps
        end_steps = start_steps + total_steps
        logging.info(f"Decay_step:{decay_step}")
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps, decay_step=decay_step)

        scaler = GradScaler() if args.precision == "amp" else None
        if args.online_learning and not args.online_cache:
            OL = OnlineLearner('test', args, max_n = 5)
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()

        loss_img = loss_img.cuda(args.gpu)
        loss_txt = loss_txt.cuda(args.gpu)
        transform = _build_transform(224)

    # Make inference for texts
    if args.extract_text_feats:
        print('Make inference for texts...')
        if args.text_feat_output_path is None:
            args.text_feat_output_path = "{}.txt_feat.jsonl".format(args.text_data[:-6])
        write_cnt = 0

        if not args.online_learning:
            args.max_epochs = 1
        with open(args.text_feat_output_path, "w") as fout:
            dataloader = text_data.dataloader
            for epoch in range(0, args.max_epochs):
                img_no = (epoch % 5) + 1
                for i, batch in enumerate(dataloader):
                    
                    text_ids, texts, raw_text = batch
                    texts = texts.cuda(args.gpu, non_blocking=True)
                    
                    if args.online_learning:
                        step = num_batches * epoch + i
                        online_imgs = []
                        if not args.online_cache:
                            OL.functions(raw_text)
                        model.train()

                        scheduler(step)

                        optimizer.zero_grad()
                        for line in raw_text:
                            line = line.strip()
                            dir = line.replace(" ", "")
                            dir = re.sub('[{}]'.format(punctuation),"", dir)
                            cnt = 0
                            if not args.online_cache:
                                while not (os.path.exists('test' + f"/{dir}/{img_no}.jpg") and Path('test' + f"/{dir}/{img_no}.jpg").stat().st_size >= 500):
                                    cnt += 1
                                    time.sleep(1)
                                    if cnt > 20:
                                        with open("mark_log/mark_test.txt", "a") as markfile:
                                            markfile.write(f"{dir}: 0")
                                        try:
                                            if os.path.exists('test' + f"/{dir}/0_{img_no}.jpg"):
                                                os.remove('test' + f"/{dir}/0_{img_no}.jpg")
                                        except:
                                            pass
                                        break
                            try:
                                img = Image.open('test' + f"/{dir}/{img_no}.jpg") # 访问图片路径
                                image = transform(img)
                                with open("mark_log/mark_test.txt", "a") as markfile:
                                    markfile.write(f"{dir}: trans")
                            except:
                                img = Image.open("assets/fail.jpg")
                                image = transform(img)

                            online_imgs.append(image)

                        online_imgs = torch.stack(online_imgs, dim=0)
                        online_imgs = online_imgs.cuda(args.gpu, non_blocking=True)

                        if args.precision == "amp":
                            with autocast():
                                total_loss, acc = get_loss(model, online_imgs, texts, loss_img, loss_txt, args)
                                scaler.scale(total_loss).backward()
                                scaler.step(optimizer)
                            scaler.update()

                        else:
                            total_loss, acc = get_loss(model, online_imgs, texts, loss_img, loss_txt, args)
                            total_loss.backward()
                            optimizer.step()

                        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

                        logging.info(
                            f"Global Steps: {step + 1}/{args.max_steps} | " +
                            f"Loss: {total_loss.item():.6f} | " +
                            (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                            (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                            f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                            f"logit_scale: {model.logit_scale.data:.3f} | "
                        )

                    if epoch == args.max_epochs - 1:
                        model.eval()
                        with torch.no_grad():
                            text_features = model(None, texts)
                            text_features /= text_features.norm(dim=-1, keepdim=True)
                            for text_id, text_feature in zip(text_ids.tolist(), text_features.tolist()):
                                fout.write("{}\n".format(json.dumps({"text_id": text_id, "feature": text_feature})))
                                write_cnt += 1


        print('{} text features are stored in {}'.format(write_cnt, args.text_feat_output_path))
    if args.online_learning:
        save_path = os.path.join(args.checkpoint_path, f"epoch{start_epoch}_online_{args.max_epochs}.pt")
        torch.save(
            {
                "epoch": start_epoch,
                "step": end_steps,
                "name": name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer_state,
            },
            save_path,
        )

    # Make inference for images
    if args.extract_image_feats:
        print('Make inference for images...')
        if args.image_feat_output_path is None:
            # by default, we store the image features under the same directory with the text features
            args.image_feat_output_path = "{}.img_feat.jsonl".format(args.text_data.replace("_texts.jsonl", "_imgs"))
        write_cnt = 0
        with open(args.image_feat_output_path, "w") as fout:
            model.eval()
            dataloader = img_data.dataloader
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    image_ids, images = batch
                    images = images.cuda(args.gpu, non_blocking=True)
                    image_features = model(images, None)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    for image_id, image_feature in zip(image_ids.tolist(), image_features.tolist()):
                        fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                        write_cnt += 1
        print('{} image features are stored in {}'.format(write_cnt, args.image_feat_output_path))

    print("Done!")