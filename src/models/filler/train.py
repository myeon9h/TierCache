'''
Train a literal filler with a selective cross entropy.
'''

import os
import json
import random
import argparse
import math
import re
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

SPECIAL_SEP = "<SEP>"
SPECIAL_MASK = "[LITERAL]"

# ===== utils =====
def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def preprocess_pred_literals(pred_literals: list, remove_target_tokens=None):
    if remove_target_tokens is None:
        remove_target_tokens = ["UNK"]

    preprocessed_pred_literals = []

    for literals_str in pred_literals:
        for special_token in remove_target_tokens:
            if special_token is not None:
                literals_str = literals_str.replace(special_token, "")
        literals_str = literals_str.strip()

        matches = re.findall(
            r"<extra_id_(\d+)>\s*(.*?)(?=\s*<extra_id_\d+>|$)",
            literals_str,
            flags=re.DOTALL,
        )

        matches = sorted(
            [(int(idx), lit.strip()) for idx, lit in matches],
            key=lambda x: x[0],
        )

        literals = [lit for _, lit in matches if lit != ""]
        preprocessed_pred_literals.append(literals)

    return preprocessed_pred_literals


def get_amp_settings(mode: str, is_train: bool):
    amp_mode = str(mode).lower()

    if amp_mode == "bf16":
        if not torch.cuda.is_available():
            raise RuntimeError("bf16 requires CUDA.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support bf16.")
        amp_enabled = True
        amp_dtype = torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", enabled=False) if is_train else None

    elif amp_mode == "fp16":
        if not torch.cuda.is_available():
            raise RuntimeError("fp16 requires CUDA.")
        amp_enabled = True
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=True) if is_train else None

    elif amp_mode in ["fp32", "none", "off", "false"]:
        amp_enabled = False
        amp_dtype = None
        scaler = torch.amp.GradScaler("cuda", enabled=False) if is_train else None

    else:
        raise ValueError(
            f"Unsupported AMP mode: {amp_mode}. "
            f"Use one of ['bf16', 'fp16', 'fp32']."
        )

    return amp_mode, amp_enabled, amp_dtype, scaler


# ===== dataset =====
class LiteralFillDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tok: T5Tokenizer,
        max_in: int = 384,
        max_tgt: int = 128,
    ):
        with open(data_path, encoding="utf-8") as f:
            data = [x for x in json.load(f) if len(x["sql"]["literals"]) > 0]

        self.samples = []
        self.tok = tok

        for ex in data:
            template = ex["sql"]["semi_template"]
            literals = ex["sql"]["literals"]
            num_slots = template.count(SPECIAL_MASK)

            if num_slots != len(literals):
                continue

            slot_tokens = " ".join([f"<extra_id_{i}>" for i in range(num_slots)])

            src = (
                f"fill literals in template {SPECIAL_SEP} "
                f"question: {ex['question']} {ex.get('evidence', '').strip()} {SPECIAL_SEP} "
                f"template: {template} {SPECIAL_SEP} "
                f"slots: {slot_tokens}"
            )

            tgt = " ".join(
                [f"<extra_id_{i}> {lit}" for i, lit in enumerate(literals)]
            )

            src_ids = tok(src, truncation=True, max_length=max_in)["input_ids"]
            tgt_ids = tok(tgt, truncation=True, max_length=max_tgt)["input_ids"]
            self.samples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch, tok):
    src, tgt = zip(*batch)

    src_pad = tok.pad(
        {"input_ids": list(map(torch.tensor, src))},
        return_tensors="pt",
    )
    tgt_pad = tok.pad(
        {"input_ids": list(map(torch.tensor, tgt))},
        return_tensors="pt",
    )["input_ids"]

    tgt_pad[tgt_pad == tok.pad_token_id] = -100
    src_pad["labels"] = tgt_pad
    return src_pad


def build_collate_fn(tok: T5Tokenizer):
    return partial(collate, tok=tok)


# ===== eval =====
def evaluate_dev(
    model,
    tok,
    dev_ds,
    batch_size,
    rank,
    max_output_length,
    eval_amp_enabled,
    eval_amp_dtype,
):
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=build_collate_fn(tok),
    )

    correct, n = 0, 0

    with torch.no_grad():
        for batch in dev_loader:
            batch = to_device(batch, rank)

            with torch.amp.autocast(
                "cuda",
                dtype=eval_amp_dtype,
                enabled=eval_amp_enabled,
            ):
                gens = model.module.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=max_output_length,
                )

            pred = tok.batch_decode(gens, skip_special_tokens=False)
            gold = tok.batch_decode(
                torch.where(batch["labels"] == -100, tok.pad_token_id, batch["labels"]),
                skip_special_tokens=False,
            )

            pred = preprocess_pred_literals(
                pred,
                remove_target_tokens=[tok.eos_token, tok.pad_token, tok.unk_token],
            )
            gold = preprocess_pred_literals(
                gold,
                remove_target_tokens=[tok.eos_token, tok.pad_token, tok.unk_token],
            )

            correct += sum(p == g for p, g in zip(pred, gold))
            n += len(pred)

    return correct / n if n > 0 else 0.0


# ===== training =====
def train(rank, world_size, config):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=60),
    )
    torch.cuda.set_device(rank)

    writer = None
    if rank == 0:
        writer = SummaryWriter(
            log_dir=os.path.join(config["out_model_dir_path"], "logs")
        )

    # AMP settings
    train_amp_mode, train_amp_enabled, train_amp_dtype, scaler = get_amp_settings(
        config.get("train_amp_dtype", "fp32"),
        is_train=True,
    )
    eval_amp_mode, eval_amp_enabled, eval_amp_dtype, _ = get_amp_settings(
        config.get("eval_amp_dtype", "fp32"),
        is_train=False,
    )

    if rank == 0:
        print(
            f"[AMP] train={train_amp_mode} "
            f"(enabled={train_amp_enabled}, dtype={train_amp_dtype}) | "
            f"eval={eval_amp_mode} "
            f"(enabled={eval_amp_enabled}, dtype={eval_amp_dtype})"
        )

    # tokenizer
    tok = T5Tokenizer.from_pretrained(
        config["pretrained_model"],
        use_fast=True,
        legacy=False,
    )

    extra = []
    if SPECIAL_SEP not in tok.get_vocab():
        extra.append(SPECIAL_SEP)
    if SPECIAL_MASK not in tok.get_vocab():
        extra.append(SPECIAL_MASK)
    if extra:
        tok.add_special_tokens({"additional_special_tokens": extra})

    # datasets
    train_ds = LiteralFillDataset(
        data_path=config["train_file_path"],
        tok=tok,
        max_in=config["max_input_length"],
        max_tgt=config["max_output_length"],
    )

    dev_ds_ehrsql = LiteralFillDataset(
        data_path=config["dev_ehrsql_file_path"],
        tok=tok,
        max_in=config["max_input_length"],
        max_tgt=config["max_output_length"],
    )

    dev_ds_sb = LiteralFillDataset(
        data_path=config["dev_sb_file_path"],
        tok=tok,
        max_in=config["max_input_length"],
        max_tgt=config["max_output_length"],
    )

    dev_ds_bird = LiteralFillDataset(
        data_path=config["dev_bird_file_path"],
        tok=tok,
        max_in=config["max_input_length"],
        max_tgt=config["max_output_length"],
    )

    train_loader = DataLoader(
        train_ds,
        sampler=DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        ),
        batch_size=config["batch_size"],
        num_workers=2,
        pin_memory=True,
        collate_fn=build_collate_fn(tok),
    )

    # model
    model = T5ForConditionalGeneration.from_pretrained(
        config["pretrained_model"],
        use_safetensors=True,
    )
    if extra:
        model.resize_token_embeddings(len(tok))

    model.cuda(rank)
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
    )

    # optimizer / scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    steps_per_epoch = math.ceil(
        len(train_loader) / config["gradient_accumulation_steps"]
    )
    total_steps = steps_per_epoch * config["epochs"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.06),
        num_training_steps=total_steps,
    )

    # train loop
    for epoch in range(1, config["epochs"] + 1):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()

        accum = config["gradient_accumulation_steps"]

        for step, batch in enumerate(train_loader, 1):
            batch = to_device(batch, rank)

            with torch.amp.autocast(
                "cuda",
                dtype=train_amp_dtype,
                enabled=train_amp_enabled,
            ):
                raw_loss = model(**batch).loss
                loss = raw_loss / accum

            scaler.scale(loss).backward()

            if step % accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if rank == 0 and step % 20 == 0:
                global_step = (epoch - 1) * len(train_loader) + step
                writer.add_scalar("Loss/train", raw_loss.item(), global_step)
                print(f"Epoch {epoch}, step {step}, Loss: {raw_loss.item():.4f}")

        dist.barrier()

        # eval & save
        if (epoch == 1 or epoch % 2 == 0) and rank == 0:
            model.eval()

            ehrsql_em = evaluate_dev(
                model=model,
                tok=tok,
                dev_ds=dev_ds_ehrsql,
                batch_size=config["batch_size"],
                rank=rank,
                max_output_length=config["max_output_length"],
                eval_amp_enabled=eval_amp_enabled,
                eval_amp_dtype=eval_amp_dtype,
            )
            writer.add_scalar("[EHRSQL] Eval/Dev EM", ehrsql_em, epoch)
            print(f"Epoch {epoch}  EHRSQL-Dev EM {ehrsql_em:.2%}")

            sb_em = evaluate_dev(
                model=model,
                tok=tok,
                dev_ds=dev_ds_sb,
                batch_size=config["batch_size"],
                rank=rank,
                max_output_length=config["max_output_length"],
                eval_amp_enabled=eval_amp_enabled,
                eval_amp_dtype=eval_amp_dtype,
            )
            writer.add_scalar("[ScienceBenchmark] Eval/Dev EM", sb_em, epoch)
            print(f"Epoch {epoch}  ScienceBenchmark-Dev EM {sb_em:.2%}")

            bird_em = evaluate_dev(
                model=model,
                tok=tok,
                dev_ds=dev_ds_bird,
                batch_size=config["batch_size"],
                rank=rank,
                max_output_length=config["max_output_length"],
                eval_amp_enabled=eval_amp_enabled,
                eval_amp_dtype=eval_amp_dtype,
            )
            writer.add_scalar("[BIRD] Eval/Dev EM", bird_em, epoch)
            print(f"Epoch {epoch}  BIRD-Dev EM {bird_em:.2%}")

            out_model_dir_path = os.path.join(
                config["out_model_dir_path"],
                f"ep{epoch}",
            )
            os.makedirs(out_model_dir_path, exist_ok=True)
            model.module.save_pretrained(out_model_dir_path, safe_serialization=True)
            tok.save_pretrained(out_model_dir_path)

        dist.barrier()

    if rank == 0:
        writer.close()

    dist.destroy_process_group()


# ===== main =====
if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        help="Path to training config JSON file",
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices found.")

    config = load_config(args.config_path)
    set_seed(config["seed"])

    mp.spawn(
        train,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )