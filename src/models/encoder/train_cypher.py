#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import socket
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# =========================================================
# Utils
# =========================================================
def safe_strip(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_json_or_jsonl(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    if raw[0] == "[":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {file_path}")
        return data

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_json(data: Dict[str, Any], path: str):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def set_ddp_env(rank: int, world_size: int, local_rank: int, master_addr: str, master_port: int):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)


def init_distributed() -> Tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Distributed environment variables are missing.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=1800),
    )
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_mean_scalar(value: float, device: torch.device) -> float:
    t = torch.tensor(value, dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / dist.get_world_size()
    return float(t.item())


def build_runtime_config(raw_config: Dict[str, Any], world_size: int) -> Dict[str, Any]:
    required_keys = [
        "train_file_path",
        "dev_file_path",
        "out_model_dir_path",
        "backbone",
        "max_nlq_length",
        "batch_size",
        "epochs",
        "samples_per_label",
        "num_explicit_negatives",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "loss_temperature",
        "gradient_accumulation_steps",
        "seed",
    ]
    for k in required_keys:
        if k not in raw_config:
            raise KeyError(f"Missing required config key: {k}")

    cpu_count = os.cpu_count() or 8
    num_workers = min(8, max(0, cpu_count // max(world_size, 1)))

    config = dict(raw_config)
    config["pooling"] = str(raw_config.get("pooling", "cls"))
    config["l2_norm"] = bool(raw_config.get("l2_norm", True))
    config["train_amp_dtype"] = str(raw_config.get("train_amp_dtype", "bf16"))
    config["eval_amp_dtype"] = str(raw_config.get("eval_amp_dtype", "fp32"))
    config["gradient_checkpointing"] = bool(raw_config.get("gradient_checkpointing", True))
    config["drop_singleton_labels"] = bool(raw_config.get("drop_singleton_labels", True))
    config["max_grad_norm"] = float(raw_config.get("max_grad_norm", 1.0))
    config["log_interval"] = int(raw_config.get("log_interval", 20))
    config["num_workers"] = int(raw_config.get("num_workers", num_workers))
    config["eval_batch_size"] = int(raw_config.get("eval_batch_size", max(64, int(config["batch_size"]) * 4)))
    config["group_loss_weight"] = float(raw_config.get("group_loss_weight", 0.3))
    return config


# =========================================================
# AMP
# =========================================================
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
        raise ValueError(f"Unsupported AMP mode: {amp_mode}. Use one of ['bf16', 'fp16', 'fp32'].")

    return amp_mode, amp_enabled, amp_dtype, scaler


# =========================================================
# Dataset
# =========================================================
class TrainDataset(Dataset):
    def __init__(self, file_path: str, num_explicit_negatives: int = 0, drop_singleton_labels: bool = True):
        rows = load_json_or_jsonl(file_path)

        processed = []
        label_counter = Counter()

        for row in rows:
            encoder_input = safe_strip(row.get("encoder_input", ""))
            if not encoder_input:
                encoder_input = f"{safe_strip(row.get('question', ''))} {safe_strip(row.get('evidence', ''))}".strip()

            item = {
                "id": int(row.get("id", -1)),
                "source": safe_strip(row.get("source", "")),
                "db": safe_strip(row.get("db", "")),
                "encoder_input": encoder_input,
                "cypher_semi_template_label": int(row["cypher_semi_template_label"]),
                "intent_group_label": int(row["intent_group_label"]),
                "hard_negatives": [safe_strip(x) for x in row.get("hard_negatives", []) if safe_strip(x)],
            }

            if num_explicit_negatives > 0:
                item["hard_negatives"] = item["hard_negatives"][:num_explicit_negatives]
            else:
                item["hard_negatives"] = []

            processed.append(item)
            label_counter[item["cypher_semi_template_label"]] += 1

        if drop_singleton_labels:
            processed = [x for x in processed if label_counter[x["cypher_semi_template_label"]] >= 2]

        self.rows = processed
        self.cypher_labels = [x["cypher_semi_template_label"] for x in self.rows]
        self.group_labels = [x["intent_group_label"] for x in self.rows]

        self.label_to_indices = defaultdict(list)
        for idx, x in enumerate(self.rows):
            self.label_to_indices[x["cypher_semi_template_label"]].append(idx)

        self.valid_labels = sorted([label for label, inds in self.label_to_indices.items() if len(inds) >= 2])

        if len(self.rows) == 0:
            raise ValueError("No training examples available after preprocessing/filtering.")
        if len(self.valid_labels) == 0:
            raise ValueError("No cypher_semi_template_label with at least 2 examples. SupCon needs positives.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


class EvalDataset(Dataset):
    def __init__(self, file_path: str):
        rows = load_json_or_jsonl(file_path)
        processed = []
        for row in rows:
            encoder_input = safe_strip(row.get("encoder_input", ""))
            if not encoder_input:
                encoder_input = f"{safe_strip(row.get('question', ''))} {safe_strip(row.get('evidence', ''))}".strip()

            processed.append({
                "id": int(row.get("id", -1)),
                "source": safe_strip(row.get("source", "")),
                "db": safe_strip(row.get("db", "")),
                "encoder_input": encoder_input,
                "cypher_semi_template_label": int(row["cypher_semi_template_label"]),
                "intent_group_label": int(row["intent_group_label"]),
            })
        self.rows = processed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


# =========================================================
# Label-balanced sampler
# =========================================================
class DistributedTemplateBatchSampler(Sampler[List[int]]):
    """
    Global batch:
      - samples labels without replacement if possible
      - each label contributes `samples_per_label` examples
      - global batch is shuffled and sharded to each rank
    """
    def __init__(
        self,
        label_to_indices: Dict[int, List[int]],
        per_device_batch_size: int,
        samples_per_label: int,
        world_size: int,
        rank: int,
        steps_per_epoch: int,
        seed: int = 42,
    ):
        self.label_to_indices = {k: list(v) for k, v in label_to_indices.items() if len(v) >= 2}
        self.labels = sorted(self.label_to_indices.keys())

        if len(self.labels) == 0:
            raise ValueError("No valid labels with >=2 examples.")

        self.per_device_batch_size = per_device_batch_size
        self.samples_per_label = samples_per_label
        self.world_size = world_size
        self.rank = rank
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0

        self.global_batch_size = per_device_batch_size * world_size
        if self.global_batch_size % self.samples_per_label != 0:
            raise ValueError(
                f"global_batch_size={self.global_batch_size} must be divisible by "
                f"samples_per_label={self.samples_per_label}"
            )

        self.labels_per_global_batch = self.global_batch_size // self.samples_per_label

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _sample_labels(self, rng: random.Random) -> List[int]:
        if len(self.labels) >= self.labels_per_global_batch:
            return rng.sample(self.labels, self.labels_per_global_batch)
        return [rng.choice(self.labels) for _ in range(self.labels_per_global_batch)]

    def _sample_examples_for_label(self, label: int, rng: random.Random) -> List[int]:
        pool = self.label_to_indices[label]
        if len(pool) >= self.samples_per_label:
            return rng.sample(pool, self.samples_per_label)

        chosen = list(pool)
        while len(chosen) < self.samples_per_label:
            chosen.append(rng.choice(pool))
        rng.shuffle(chosen)
        return chosen

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        for _ in range(self.steps_per_epoch):
            selected_labels = self._sample_labels(rng)

            global_indices = []
            for label in selected_labels:
                global_indices.extend(self._sample_examples_for_label(label, rng))

            rng.shuffle(global_indices)
            start = self.rank * self.per_device_batch_size
            end = start + self.per_device_batch_size
            yield global_indices[start:end]


# =========================================================
# Model
# =========================================================
class BGEEncoder(nn.Module):
    def __init__(
        self,
        backbone_path: str,
        pooling: str = "cls",
        l2_norm: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone_path, trust_remote_code=True)

        # Remove unused pooler if present.
        if hasattr(self.backbone, "pooler") and self.backbone.pooler is not None:
            self.backbone.pooler = None

        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            try:
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.backbone.gradient_checkpointing_enable()

        self.pooling = pooling
        self.l2_norm = l2_norm

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            emb = hidden[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            raise ValueError(f"Unsupported pooling={self.pooling}. Use 'cls' or 'mean'.")

        if self.l2_norm:
            emb = F.normalize(emb, p=2, dim=-1)

        return emb


# =========================================================
# Distributed gather
# =========================================================
class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if not dist.is_initialized():
            return (x,)
        outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(outputs, x)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        if not dist.is_initialized():
            return grads[0]
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    gathered = GatherLayer.apply(x)
    return torch.cat(gathered, dim=0)


def gather_tensor_no_grad(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    xs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(xs, x)
    return torch.cat(xs, dim=0)


# =========================================================
# Collators
# =========================================================
class TrainCollator:
    def __init__(self, tokenizer, max_length: int, num_explicit_negatives: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_explicit_negatives = num_explicit_negatives

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [safe_strip(x["encoder_input"]) for x in batch]
        cypher_labels = torch.tensor([int(x["cypher_semi_template_label"]) for x in batch], dtype=torch.long)
        group_labels = torch.tensor([int(x["intent_group_label"]) for x in batch], dtype=torch.long)

        anchor_tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        neg_texts_flat = []
        neg_counts = []
        if self.num_explicit_negatives > 0:
            for x in batch:
                negs = [safe_strip(n) for n in x.get("hard_negatives", []) if safe_strip(n)]
                negs = negs[:self.num_explicit_negatives]
                neg_counts.append(len(negs))
                neg_texts_flat.extend(negs)
        else:
            neg_counts = [0 for _ in batch]

        if len(neg_texts_flat) > 0:
            neg_tok = self.tokenizer(
                neg_texts_flat,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            neg_tok = None

        return {
            "anchor_inputs": anchor_tok,
            "cypher_labels": cypher_labels,
            "group_labels": group_labels,
            "neg_inputs": neg_tok,
            "neg_counts": neg_counts,
        }


class EvalCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [safe_strip(x["encoder_input"]) for x in batch]
        cypher_labels = torch.tensor([int(x["cypher_semi_template_label"]) for x in batch], dtype=torch.long)
        group_labels = torch.tensor([int(x["intent_group_label"]) for x in batch], dtype=torch.long)
        ids = torch.tensor([int(x["id"]) for x in batch], dtype=torch.long)
        dbs = [safe_strip(x["db"]) for x in batch]

        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "inputs": tok,
            "cypher_labels": cypher_labels,
            "group_labels": group_labels,
            "ids": ids,
            "dbs": dbs,
        }


# =========================================================
# Losses
# =========================================================
def compute_exact_supcon_with_group_mask_and_explicit_negatives(
    local_emb: torch.Tensor,
    local_cypher_labels: torch.Tensor,
    local_group_labels: torch.Tensor,
    rank: int,
    temperature: float,
    explicit_neg_emb: Optional[torch.Tensor] = None,
    neg_counts: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Exact loss:
      positive = same cypher_semi_template_label
      negative = all except:
        - self
        - same intent_group_label but different cypher_semi_template_label
      explicit negatives are appended only to denominator
    """
    device = local_emb.device
    B = local_emb.size(0)

    global_emb = gather_with_grad(local_emb)
    global_cypher_labels = gather_tensor_no_grad(local_cypher_labels.to(device))
    global_group_labels = gather_tensor_no_grad(local_group_labels.to(device))
    G = global_emb.size(0)

    logits_global = torch.matmul(local_emb, global_emb.T) / temperature

    self_mask = torch.zeros((B, G), device=device, dtype=torch.bool)
    start = rank * B
    self_cols = torch.arange(B, device=device) + start
    self_mask[torch.arange(B, device=device), self_cols] = True

    positive_mask = (local_cypher_labels.unsqueeze(1) == global_cypher_labels.unsqueeze(0)) & (~self_mask)

    ambiguous_same_group_diff_cypher = (
        (local_group_labels.unsqueeze(1) == global_group_labels.unsqueeze(0))
        & (local_cypher_labels.unsqueeze(1) != global_cypher_labels.unsqueeze(0))
    )

    denom_mask = (~self_mask) & (~ambiguous_same_group_diff_cypher)
    logits_batch = logits_global.masked_fill(~denom_mask, float("-inf"))

    if explicit_neg_emb is not None and explicit_neg_emb.numel() > 0 and neg_counts is not None:
        neg_splits = list(neg_counts)
        max_k = max(neg_splits) if len(neg_splits) > 0 else 0
        if max_k > 0:
            neg_chunks = torch.split(explicit_neg_emb, neg_splits, dim=0)
            neg_logits = local_emb.new_full((B, max_k), float("-inf"))
            for i, neg_emb_i in enumerate(neg_chunks):
                if neg_emb_i.numel() == 0:
                    continue
                scores_i = torch.matmul(local_emb[i:i+1], neg_emb_i.T).squeeze(0) / temperature
                neg_logits[i, :scores_i.size(0)] = scores_i
            denom_all = torch.cat([logits_batch, neg_logits], dim=1)
        else:
            denom_all = logits_batch
    else:
        denom_all = logits_batch

    denom_logsumexp = torch.logsumexp(denom_all, dim=1, keepdim=True)
    log_prob = logits_global - denom_logsumexp

    positive_mask_f = positive_mask.float()
    num_pos = positive_mask_f.sum(dim=1)
    valid = num_pos > 0

    per_anchor_loss = -(log_prob * positive_mask_f).sum(dim=1) / num_pos.clamp(min=1.0)
    loss = per_anchor_loss[valid].mean() if valid.any() else local_emb.sum() * 0.0

    with torch.no_grad():
        stats = {
            "valid_anchor_ratio": float(valid.float().mean().item()),
            "avg_positives_per_anchor": float(num_pos.mean().item()),
        }
    return loss, stats


def compute_group_supcon(
    local_emb: torch.Tensor,
    local_group_labels: torch.Tensor,
    rank: int,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Group loss:
      positive = same intent_group_label
      negative = all except self
    """
    device = local_emb.device
    B = local_emb.size(0)

    global_emb = gather_with_grad(local_emb)
    global_group_labels = gather_tensor_no_grad(local_group_labels.to(device))
    G = global_emb.size(0)

    logits_global = torch.matmul(local_emb, global_emb.T) / temperature

    self_mask = torch.zeros((B, G), device=device, dtype=torch.bool)
    start = rank * B
    self_cols = torch.arange(B, device=device) + start
    self_mask[torch.arange(B, device=device), self_cols] = True

    positive_mask = (local_group_labels.unsqueeze(1) == global_group_labels.unsqueeze(0)) & (~self_mask)
    logits_batch = logits_global.masked_fill(self_mask, float("-inf"))

    denom_logsumexp = torch.logsumexp(logits_batch, dim=1, keepdim=True)
    log_prob = logits_global - denom_logsumexp

    positive_mask_f = positive_mask.float()
    num_pos = positive_mask_f.sum(dim=1)
    valid = num_pos > 0

    per_anchor_loss = -(log_prob * positive_mask_f).sum(dim=1) / num_pos.clamp(min=1.0)
    loss = per_anchor_loss[valid].mean() if valid.any() else local_emb.sum() * 0.0

    with torch.no_grad():
        stats = {
            "valid_anchor_ratio": float(valid.float().mean().item()),
            "avg_positives_per_anchor": float(num_pos.mean().item()),
        }
    return loss, stats


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def encode_dataset_distributed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    model.eval()
    local_embeddings = []
    local_cypher_labels = []
    local_group_labels = []
    local_dbs = []

    for batch in loader:
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch["inputs"].items()}
        cypher_labels = batch["cypher_labels"].cpu().numpy()
        group_labels = batch["group_labels"].cpu().numpy()
        dbs = np.array(batch["dbs"], dtype=object)

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            emb = model(**inputs)
        emb = F.normalize(emb.float(), p=2, dim=-1)

        local_embeddings.append(emb.cpu().numpy())
        local_cypher_labels.append(cypher_labels)
        local_group_labels.append(group_labels)
        local_dbs.append(dbs)

    if len(local_embeddings) == 0:
        hidden_size = unwrap_model(model).backbone.config.hidden_size
        local_embeddings = np.zeros((0, hidden_size), dtype=np.float32)
        local_cypher_labels = np.zeros((0,), dtype=np.int64)
        local_group_labels = np.zeros((0,), dtype=np.int64)
        local_dbs = np.zeros((0,), dtype=object)
    else:
        local_embeddings = np.concatenate(local_embeddings, axis=0)
        local_cypher_labels = np.concatenate(local_cypher_labels, axis=0)
        local_group_labels = np.concatenate(local_group_labels, axis=0)
        local_dbs = np.concatenate(local_dbs, axis=0)

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, (local_embeddings, local_cypher_labels, local_group_labels, local_dbs))

    if dist.get_rank() == 0:
        embs = np.concatenate([x[0] for x in gathered], axis=0)
        cypher_labels = np.concatenate([x[1] for x in gathered], axis=0)
        group_labels = np.concatenate([x[2] for x in gathered], axis=0)
        dbs = np.concatenate([x[3] for x in gathered], axis=0)
        return embs, cypher_labels, group_labels, dbs
    return None


@torch.no_grad()
def compute_same_db_recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    dbs: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 4096,
) -> Dict[str, float]:
    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = 0.0
        metrics[f"eligible_recall@{k}"] = 0.0
    metrics["eligible_query_ratio"] = 0.0

    n = embeddings.shape[0]
    if n <= 1:
        return metrics

    db_to_indices = defaultdict(list)
    for i, db in enumerate(dbs.tolist()):
        db_to_indices[db].append(i)

    label_counter_same_db = {}
    for db, inds in db_to_indices.items():
        counter = Counter(labels[inds].tolist())
        for idx in inds:
            label_counter_same_db[idx] = counter[int(labels[idx])]

    eligible_mask_np = np.array([label_counter_same_db[i] >= 2 for i in range(n)], dtype=np.bool_)
    eligible_total = int(eligible_mask_np.sum())
    metrics["eligible_query_ratio"] = eligible_total / max(n, 1)

    emb = torch.from_numpy(embeddings).float().cuda()
    lab = torch.from_numpy(labels).long().cuda()
    eligible_mask = torch.from_numpy(eligible_mask_np).to(device=emb.device, dtype=torch.bool)

    hits_all = {k: 0 for k in ks}
    hits_eligible = {k: 0 for k in ks}

    for db, inds in db_to_indices.items():
        if len(inds) <= 1:
            continue

        local_emb = emb[inds]
        local_lab = lab[inds]
        local_eligible = eligible_mask[inds]

        m = len(inds)
        max_k = min(max(ks), m - 1)

        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            q = local_emb[start:end]
            sims = torch.matmul(q, local_emb.T)
            row_idx = torch.arange(end - start, device=sims.device)
            sims[row_idx, start + row_idx] = -1e9

            topk_idx = torch.topk(sims, k=max_k, dim=1, largest=True).indices
            q_labels = local_lab[start:end].unsqueeze(1)
            retrieved_labels = local_lab[topk_idx]
            matches = (retrieved_labels == q_labels)
            eligible_chunk = local_eligible[start:end]

            for k in ks:
                eff_k = min(k, max_k)
                hit_k = matches[:, :eff_k].any(dim=1)
                hits_all[k] += int(hit_k.sum().item())
                if eligible_chunk.any():
                    hits_eligible[k] += int(hit_k[eligible_chunk].sum().item())

    for k in ks:
        metrics[f"recall@{k}"] = hits_all[k] / max(n, 1)
        metrics[f"eligible_recall@{k}"] = hits_eligible[k] / max(eligible_total, 1)

    return metrics


@torch.no_grad()
def evaluate_dev(
    model: nn.Module,
    tokenizer,
    dev_ds: EvalDataset,
    batch_size: int,
    rank: int,
    world_size: int,
    max_length: int,
    eval_amp_enabled: bool,
    eval_amp_dtype: Optional[torch.dtype],
    num_workers: int,
) -> Optional[Dict[str, Dict[str, float]]]:
    sampler = torch.utils.data.distributed.DistributedSampler(
        dev_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    collator = EvalCollator(tokenizer=tokenizer, max_length=max_length)

    loader_kwargs = dict(
        dataset=dev_ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(**loader_kwargs)

    encoded = encode_dataset_distributed(
        model=model,
        loader=loader,
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        amp_enabled=eval_amp_enabled,
        amp_dtype=eval_amp_dtype,
    )

    if rank == 0:
        embeddings, cypher_labels, group_labels, dbs = encoded
        exact_metrics = compute_same_db_recall_at_k(embeddings, cypher_labels, dbs, ks=(1, 5, 10))
        group_metrics = compute_same_db_recall_at_k(embeddings, group_labels, dbs, ks=(1, 5, 10))
        return {
            "exact": exact_metrics,
            "group": group_metrics,
        }
    return None


# =========================================================
# Training
# =========================================================
def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)


def maybe_move_batch_to_device(batch_inputs: Optional[Dict[str, torch.Tensor]], device: torch.device):
    if batch_inputs is None:
        return None
    return {k: v.to(device, non_blocking=True) for k, v in batch_inputs.items()}


def train_one_epoch(
    model: nn.Module,
    tokenizer,
    train_ds: TrainDataset,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    config: Dict[str, Any],
    device: torch.device,
    rank: int,
    world_size: int,
    writer: Optional[SummaryWriter],
    train_amp_enabled: bool,
    train_amp_dtype: Optional[torch.dtype],
):
    per_device_batch_size = int(config["batch_size"])
    samples_per_label = int(config["samples_per_label"])
    steps_per_epoch = math.ceil(len(train_ds) / (per_device_batch_size * world_size))
    grad_accum_steps = int(config["gradient_accumulation_steps"])
    max_length = int(config["max_nlq_length"])
    num_workers = int(config["num_workers"])
    num_explicit_negatives = int(config["num_explicit_negatives"])
    group_loss_weight = float(config["group_loss_weight"])

    sampler = DistributedTemplateBatchSampler(
        label_to_indices=train_ds.label_to_indices,
        per_device_batch_size=per_device_batch_size,
        samples_per_label=samples_per_label,
        world_size=world_size,
        rank=rank,
        steps_per_epoch=steps_per_epoch,
        seed=int(config["seed"]),
    )
    sampler.set_epoch(epoch)

    loader_kwargs = dict(
        dataset=train_ds,
        batch_sampler=sampler,
        collate_fn=TrainCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            num_explicit_negatives=num_explicit_negatives,
        ),
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(**loader_kwargs)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    epoch_loss = 0.0
    running_steps = 0
    global_step_base = epoch * steps_per_epoch
    log_interval = int(config["log_interval"])

    remainder = steps_per_epoch % grad_accum_steps
    final_group_size = remainder if remainder != 0 else grad_accum_steps

    for step, batch in enumerate(loader):
        cypher_labels = batch["cypher_labels"].to(device, non_blocking=True)
        group_labels = batch["group_labels"].to(device, non_blocking=True)
        anchor_inputs = maybe_move_batch_to_device(batch["anchor_inputs"], device)
        neg_inputs = maybe_move_batch_to_device(batch["neg_inputs"], device)
        neg_counts = batch["neg_counts"]

        in_last_partial_group = (remainder != 0) and (step >= steps_per_epoch - remainder)
        accum_denom = final_group_size if in_last_partial_group else grad_accum_steps
        should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == steps_per_epoch)

        sync_context = model.no_sync() if (isinstance(model, DDP) and not should_step) else nullcontext()

        with sync_context:
            with torch.amp.autocast("cuda", enabled=train_amp_enabled, dtype=train_amp_dtype):
                anchor_emb = model(**anchor_inputs)
                neg_emb = model(**neg_inputs) if neg_inputs is not None else None

                exact_loss, exact_stats = compute_exact_supcon_with_group_mask_and_explicit_negatives(
                    local_emb=anchor_emb,
                    local_cypher_labels=cypher_labels,
                    local_group_labels=group_labels,
                    rank=rank,
                    temperature=float(config["loss_temperature"]),
                    explicit_neg_emb=neg_emb,
                    neg_counts=neg_counts,
                )

                group_loss, group_stats = compute_group_supcon(
                    local_emb=anchor_emb,
                    local_group_labels=group_labels,
                    rank=rank,
                    temperature=float(config["loss_temperature"]),
                )

                raw_loss = exact_loss + group_loss_weight * group_loss
                loss = raw_loss / accum_denom

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if should_step:
            max_grad_norm = float(config["max_grad_norm"])
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        reduced_loss = raw_loss.detach()
        if dist.is_initialized():
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss = reduced_loss / world_size

        exact_valid = reduce_mean_scalar(exact_stats["valid_anchor_ratio"], device)
        exact_avg_pos = reduce_mean_scalar(exact_stats["avg_positives_per_anchor"], device)
        group_valid = reduce_mean_scalar(group_stats["valid_anchor_ratio"], device)
        group_avg_pos = reduce_mean_scalar(group_stats["avg_positives_per_anchor"], device)

        epoch_loss += reduced_loss.item()
        running_steps += 1

        if is_main_process(rank) and ((step + 1) % log_interval == 0 or (step + 1) == steps_per_epoch):
            current_step = global_step_base + step + 1
            lr = scheduler.get_last_lr()[0]
            avg_loss = epoch_loss / max(running_steps, 1)
            writer.add_scalar("Train/Loss", avg_loss, current_step)
            writer.add_scalar("Train/LR", lr, current_step)
            writer.add_scalar("Train/ExactValidAnchorRatio", exact_valid, current_step)
            writer.add_scalar("Train/ExactAvgPositivesPerAnchor", exact_avg_pos, current_step)
            writer.add_scalar("Train/GroupValidAnchorRatio", group_valid, current_step)
            writer.add_scalar("Train/GroupAvgPositivesPerAnchor", group_avg_pos, current_step)

            print(
                f"Epoch {epoch} Step {step+1}/{steps_per_epoch} "
                f"Loss {avg_loss:.4f} LR {lr:.2e} "
                f"ExactValid {exact_valid:.3f} ExactAvgPos {exact_avg_pos:.2f} "
                f"GroupValid {group_valid:.3f} GroupAvgPos {group_avg_pos:.2f}",
                flush=True,
            )

    return epoch_loss / max(running_steps, 1)


# =========================================================
# Main training body
# =========================================================
def run_main(config_path: str):
    rank, world_size, local_rank, device = init_distributed()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        config = build_runtime_config(raw_config, world_size)

        os.makedirs(config["out_model_dir_path"], exist_ok=True)
        set_seed(int(config["seed"]) + rank)

        if is_main_process(rank):
            save_json(config, os.path.join(config["out_model_dir_path"], "used_config.json"))
            writer = SummaryWriter(log_dir=os.path.join(config["out_model_dir_path"], "tb"))
        else:
            writer = None

        train_amp_mode, train_amp_enabled, train_amp_dtype, scaler = get_amp_settings(
            config["train_amp_dtype"], is_train=True
        )
        dev_amp_mode, dev_amp_enabled, dev_amp_dtype, _ = get_amp_settings(
            config["eval_amp_dtype"], is_train=False
        )

        tokenizer = AutoTokenizer.from_pretrained(config["backbone"], trust_remote_code=True)
        model = BGEEncoder(
            backbone_path=config["backbone"],
            pooling=str(config["pooling"]),
            l2_norm=bool(config["l2_norm"]),
            gradient_checkpointing=bool(config["gradient_checkpointing"]),
        ).to(device)

        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

        train_ds = TrainDataset(
            file_path=config["train_file_path"],
            num_explicit_negatives=int(config["num_explicit_negatives"]),
            drop_singleton_labels=bool(config["drop_singleton_labels"]),
        )
        dev_ds = EvalDataset(config["dev_file_path"])

        if is_main_process(rank):
            print(f"Train examples after filtering: {len(train_ds)}")
            print(f"#valid cypher labels: {len(train_ds.valid_labels)}")
            print(f"Dev size: {len(dev_ds)}")
            print(f"Train AMP: {train_amp_mode}, Dev AMP: {dev_amp_mode}")

        optimizer = build_optimizer(
            model=model,
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )

        per_device_batch_size = int(config["batch_size"])
        steps_per_epoch = math.ceil(len(train_ds) / (per_device_batch_size * world_size))
        grad_accum_steps = int(config["gradient_accumulation_steps"])
        optimizer_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)
        total_optimizer_steps = optimizer_steps_per_epoch * int(config["epochs"])
        warmup_steps = int(total_optimizer_steps * float(config["warmup_ratio"]))

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

        best_metric = -1.0
        eval_batch_size = int(config["eval_batch_size"])
        num_workers = int(config["num_workers"])

        for epoch in range(int(config["epochs"])):
            t0 = time.time()

            train_loss = train_one_epoch(
                model=model,
                tokenizer=tokenizer,
                train_ds=train_ds,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                config=config,
                device=device,
                rank=rank,
                world_size=world_size,
                writer=writer,
                train_amp_enabled=train_amp_enabled,
                train_amp_dtype=train_amp_dtype,
            )
            dist.barrier()

            if is_main_process(rank):
                writer.add_scalar("Train/EpochLoss", train_loss, epoch)
                print(f"Epoch {epoch} Train Loss {train_loss:.4f} ({time.time()-t0:.1f}s)", flush=True)

            dev_metrics = evaluate_dev(
                model=model,
                tokenizer=tokenizer,
                dev_ds=dev_ds,
                batch_size=eval_batch_size,
                rank=rank,
                world_size=world_size,
                max_length=int(config["max_nlq_length"]),
                eval_amp_enabled=dev_amp_enabled,
                eval_amp_dtype=dev_amp_dtype,
                num_workers=num_workers,
            )
            dist.barrier()

            if is_main_process(rank):
                for level in ["exact", "group"]:
                    m = dev_metrics[level]
                    writer.add_scalar(f"[Dev] Eval/{level}_Recall@1", m["recall@1"], epoch)
                    writer.add_scalar(f"[Dev] Eval/{level}_Recall@5", m["recall@5"], epoch)
                    writer.add_scalar(f"[Dev] Eval/{level}_Recall@10", m["recall@10"], epoch)

                    writer.add_scalar(f"[Dev] Eval/{level}_EligibleRecall@1", m["eligible_recall@1"], epoch)
                    writer.add_scalar(f"[Dev] Eval/{level}_EligibleRecall@5", m["eligible_recall@5"], epoch)
                    writer.add_scalar(f"[Dev] Eval/{level}_EligibleRecall@10", m["eligible_recall@10"], epoch)
                    writer.add_scalar(f"[Dev] Eval/{level}_EligibleQueryRatio", m["eligible_query_ratio"], epoch)

                print(
                    f"Epoch {epoch} Dev | "
                    f"Exact R@1 {dev_metrics['exact']['recall@1']:.2%} "
                    f"R@5 {dev_metrics['exact']['recall@5']:.2%} "
                    f"R@10 {dev_metrics['exact']['recall@10']:.2%} "
                    f"EligibleR@1 {dev_metrics['exact']['eligible_recall@1']:.2%} "
                    f"EligibleR@5 {dev_metrics['exact']['eligible_recall@5']:.2%} "
                    f"EligibleR@10 {dev_metrics['exact']['eligible_recall@10']:.2%} "
                    f"(EligibleRatio {dev_metrics['exact']['eligible_query_ratio']:.2%}) | "
                    f"Group R@1 {dev_metrics['group']['recall@1']:.2%} "
                    f"R@5 {dev_metrics['group']['recall@5']:.2%} "
                    f"R@10 {dev_metrics['group']['recall@10']:.2%} "
                    f"EligibleR@1 {dev_metrics['group']['eligible_recall@1']:.2%} "
                    f"EligibleR@5 {dev_metrics['group']['eligible_recall@5']:.2%} "
                    f"EligibleR@10 {dev_metrics['group']['eligible_recall@10']:.2%} "
                    f"(EligibleRatio {dev_metrics['group']['eligible_query_ratio']:.2%})",
                    flush=True,
                )

                # reranker가 없으므로 best checkpoint는 top-1 조건부 성능 기준
                score = dev_metrics["exact"]["eligible_recall@1"]
                writer.add_scalar("Eval/ExactEligibleRecall@1", score, epoch)

                ckpt_dir = os.path.join(config["out_model_dir_path"], f"epoch_{epoch}")
                os.makedirs(ckpt_dir, exist_ok=True)

                unwrap_model(model).backbone.save_pretrained(ckpt_dir, safe_serialization=True)
                tokenizer.save_pretrained(ckpt_dir)
                save_json(
                    {
                        "pooling": config["pooling"],
                        "l2_norm": bool(config["l2_norm"]),
                        "max_nlq_length": int(config["max_nlq_length"]),
                    },
                    os.path.join(ckpt_dir, "wrapper_config.json"),
                )
                save_json(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "dev": dev_metrics,
                        "best_metric_name": "exact_eligible_recall@1",
                        "exact_eligible_recall@1": score,
                    },
                    os.path.join(ckpt_dir, "metrics.json"),
                )

                if score > best_metric:
                    best_metric = score
                    best_dir = os.path.join(config["out_model_dir_path"], "best_checkpoint")
                    os.makedirs(best_dir, exist_ok=True)
                    unwrap_model(model).backbone.save_pretrained(best_dir, safe_serialization=True)
                    tokenizer.save_pretrained(best_dir)
                    save_json(
                        {
                            "pooling": config["pooling"],
                            "l2_norm": bool(config["l2_norm"]),
                            "max_nlq_length": int(config["max_nlq_length"]),
                        },
                        os.path.join(best_dir, "wrapper_config.json"),
                    )
                    save_json(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "dev": dev_metrics,
                            "best_metric_name": "exact_eligible_recall@1",
                            "exact_eligible_recall@1": score,
                        },
                        os.path.join(best_dir, "metrics.json"),
                    )
                    print(
                        f"New best checkpoint at epoch {epoch} "
                        f"with exact eligible R@1 = {score:.2%}",
                        flush=True,
                    )

        if is_main_process(rank):
            writer.close()

    finally:
        cleanup_distributed()


# =========================================================
# Spawn helpers
# =========================================================
def distributed_worker(local_rank: int, world_size: int, config_path: str, master_addr: str, master_port: int):
    set_ddp_env(
        rank=local_rank,
        world_size=world_size,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
    )
    run_main(config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    # torchrun path
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        run_main(args.config_path)
        return

    # Auto-spawn path
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 0:
        raise RuntimeError("No visible CUDA devices found.")

    master_addr = "127.0.0.1"
    master_port = find_free_port()

    if visible_gpu_count == 1:
        set_ddp_env(
            rank=0,
            world_size=1,
            local_rank=0,
            master_addr=master_addr,
            master_port=master_port,
        )
        run_main(args.config_path)
    else:
        mp.spawn(
            distributed_worker,
            nprocs=visible_gpu_count,
            args=(visible_gpu_count, args.config_path, master_addr, master_port),
            join=True,
        )


if __name__ == "__main__":
    main()