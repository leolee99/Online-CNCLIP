import os
import re
import time
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from pathlib import Path
from zhon.hanzi import punctuation

import logging

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

def is_master(args):
    return args.rank == 0

def get_loss(model, images, texts, loss_img, loss_txt, args, temp=1000):
    image_features, text_features, logit_scale = model(images, texts)
    logit_scale = logit_scale.mean()
    if args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long()
    ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)

    # print(logits_per_image)
    # logits_per_image = torch.exp(logits_per_image)
    # logits_per_text = torch.exp(logits_per_text)

    # print(logits_per_image)
    # image_prior = F.log_softmax(logits_per_image/temp, dim=0)
    # text_prior = F.log_softmax(logits_per_text/temp, dim=0)

    # print(logits_per_image)
    # logits_per_image = logits_per_image * image_prior
    # logits_per_text = logits_per_text * text_prior
    # # logits_per_image = torch.exp(logits_per_image * image_prior)
    # # logits_per_text = torch.exp(logits_per_text * text_prior)
    # print(logits_per_image)

    # logits_per_image = F.log_softmax(logits_per_image/temp, dim=1)
    # logits_per_text = F.log_softmax(logits_per_text/temp, dim=1)
    # print(logits_per_image)

    # total_loss = (
    # loss_img(logits_per_image, ground_truth)
    # + loss_txt(logits_per_text, ground_truth)
    # ) / 2

    sim_matrix_i2t = logits_per_image * F.softmax(logits_per_image/temp, dim=0)*len(logits_per_image) #With an appropriate temperature parameter, the model achieves higher performance
    logpt_i2t = F.log_softmax(sim_matrix_i2t, dim=-1)
    logpt_i2t = torch.diag(logpt_i2t)
    sim_matrix_t2i = logits_per_text * F.softmax(logits_per_text/temp, dim=0)*len(logits_per_text) #With an appropriate temperature parameter, the model achieves higher performance
    logpt_t2i = F.log_softmax(sim_matrix_t2i, dim=-1)
    logpt_t2i = torch.diag(logpt_t2i)
    # print(logpt_t2i.shape)
    # print(logpt_t2i)

    total_loss = -logpt_i2t.mean() - logpt_t2i.mean()

    acc = None
    if args.report_training_batch_acc:
        i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
        t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
        acc = {"i2t": i2t_acc, "t2i": t2i_acc}

    return total_loss, acc


def train(model, data, epoch, optimizer, scaler, scheduler, args, global_trained_steps, OL=None):
    # os.environ["WDS_EPOCH"] = str(epoch)
    
    model.train()

    dataloader, sampler = data['train'].dataloader,  data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    if sampler is not None:
        sampler.set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches
    data_iter = iter(dataloader)

    end = time.time()
    epoch_trained_steps = 0

    for i in range(global_trained_steps - num_batches_per_epoch * epoch, num_batches_per_epoch):
        batch = next(data_iter)
        step = num_batches_per_epoch * epoch + i
        # reach the args.max_steps, exit training:
        if step >= args.max_steps:
            logging.info("Stopping training due to step {} has reached max_steps {}".format(step, args.max_steps))
            return epoch_trained_steps
        scheduler(step)

        optimizer.zero_grad()

        images, texts, eos_indices, _ = batch

        images = images.cuda(args.local_device_rank, non_blocking=True)
        texts = texts.cuda(args.local_device_rank, non_blocking=True)
        eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)

        data_time = time.time() - end

        m = model.module

        # with automatic mixed precision.
        if args.precision == "amp":
            with autocast():
                total_loss, acc = get_loss(model, images, texts, loss_img, loss_txt, args)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
            scaler.update()

        else:
            total_loss, acc = get_loss(model, images, texts, loss_img, loss_txt, args)
            total_loss.backward()
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        epoch_trained_steps += 1

        if is_master(args) and ((step + 1) % args.log_interval) == 0:
            num_samples = (i + 1) * len(images) * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * (i + 1) / num_batches_per_epoch
            
            logging.info(
                f"Global Steps: {step + 1}/{args.max_steps} | " +
                f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                f"Loss: {total_loss.item():.6f} | " +
                (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                f"Data Time: {data_time:.3f}s | " +
                f"Batch Time: {batch_time:.3f}s | " +
                f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                f"logit_scale: {m.logit_scale.data:.3f} | " +
                f"Global Batch Size: {len(images) * args.world_size}"
            )

        if args.val_data is not None and args.valid_step_interval is not None and ((step + 1) % args.valid_step_interval) == 0:
            assert "val" in data, "Error: Valid dataset has not been built."
            evaluate(model, optimizer, scaler, data, epoch, args, step + 1, OL)

        if args.should_save and args.save_step_frequency > 0 and ((step + 1) % args.save_step_frequency) == 0:
            save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_{step + 1}.pt")
            t1 = time.time()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, step + 1, time.time() - t1))

            # Save the latest params
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, step + 1, time.time() - t1))
        
    return epoch_trained_steps


def evaluate(model, optimizer, scaler, data, epoch, args, steps, OL=None):

    logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, steps))

    dataloader = data['val'].dataloader
    data_iter = iter(dataloader)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    cumulative_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_i2t_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_t2i_acc = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    num_elements = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    all_image_features, all_text_features = [], []
    transform = _build_transform(224)
    for i in range(dataloader.num_batches):
        batch = next(data_iter)
        images, texts, eos_indices, raw_text = batch

        images = images.cuda(args.local_device_rank, non_blocking=True)
        texts = texts.cuda(args.local_device_rank, non_blocking=True)
        eos_indices = eos_indices.cuda(args.local_device_rank, non_blocking=True)

        if args.online_learning:
            model.train()
            optimizer.zero_grad()

            online_imgs = []

            if not args.online_cache:
                OL.functions(raw_text)
            for line in raw_text:
                line = line.strip()
                dir = line.replace(" ", "")
                dir = re.sub('[{}]'.format(punctuation),"", dir)

                if not args.online_cache:
                    cnt = 0
                    while not (os.path.exists('valid' + f"/{dir}/1.jpg") and Path('valid' + f"/{dir}/1.jpg").stat().st_size >= 500):
                        cnt += 1
                        time.sleep(1)
                        if cnt > 15:
                            with open("mark_log/mark.txt", "a") as markfile:
                                markfile.write(f"{dir}: 0")
                            try:
                                if os.path.exists('valid' + f"/{dir}/0.jpg"):
                                    os.remove('valid' + f"/{dir}/0.jpg")
                            except:
                                pass
                            break
                try:
                    img = Image.open('valid' + f"/{dir}/1.jpg") # 访问图片路径
                    image = transform(img)
                    with open("mark_log/mark.txt", "a") as markfile:
                        markfile.write(f"{dir}: trans")
                except:
                    img = Image.open("assets/fail.jpg")
                    image = transform(img)
                online_imgs.append(image)

            online_imgs = torch.stack(online_imgs, dim=0)
            online_imgs = online_imgs.cuda(args.local_device_rank, non_blocking=True)

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

        model.eval()
        with torch.no_grad():
            image_features, text_features, logit_scale = model(images, texts)
            all_image_features.append(image_features)
            all_text_features.append(text_features)
            logit_scale = logit_scale.mean()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(len(images)).long()
            ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)
            total_loss = (
                loss_img(logits_per_image, ground_truth)
                + loss_txt(logits_per_text, ground_truth)
            ) / 2

            batch_size = len(images)
            cumulative_loss += total_loss * batch_size
            num_elements += batch_size

            cumulative_i2t_acc += ((logits_per_image.argmax(-1) == ground_truth).sum()).float()
            cumulative_t2i_acc += (logits_per_text.argmax(-1) == ground_truth).sum().float()

            if (i + 1) % 10 == 0:
                logging.info("Evaluated {}/{} batches...".format(i + 1, dataloader.num_batches))
                logging.info(
                    f"Image2Text Acc: {(cumulative_i2t_acc.item() * 100)/num_elements:.2f} | " 
                    f"Text2Image Acc: {(cumulative_t2i_acc.item() * 100)/num_elements:.2f} | " )

    dist.all_reduce(cumulative_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(cumulative_i2t_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(cumulative_t2i_acc, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_elements, op=dist.ReduceOp.SUM)
    loss = cumulative_loss / num_elements
    i2t_acc = cumulative_i2t_acc / num_elements
    t2i_acc = cumulative_t2i_acc / num_elements

    assert num_elements.item() == dataloader.num_samples # sanity check

    logging.info(
        f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
        f"Valid Loss: {loss.item():.6f} | "
        f"Image2Text Acc: {i2t_acc.item() * 100:.2f} | " 
        f"Text2Image Acc: {t2i_acc.item() * 100:.2f} | " 
        f"logit_scale: {model.module.logit_scale.data:.3f} | "
        f"Valid Batch Size: {batch_size}"
    )