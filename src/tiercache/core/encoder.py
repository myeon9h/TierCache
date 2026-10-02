import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Union
import numpy as np
from numpy.typing import NDArray

class BaseEncoder:
    def __init__(self):
        pass

    def _tokenize_and_encode(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class SFREncoder(BaseEncoder):
    def __init__(
        self, 
        model_path: str, 
        device: Union[str, torch.device], 
        max_length: int = 128, 
        batch_size: int = 32
    ):
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Code-400M_R", trust_remote_code=True)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _tokenize_and_encode(self, texts: List[str]) -> NDArray[np.float32]:
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        # cls = outputs.last_hidden_state[:, 0] # [CLS] token embedding
        reps = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])  # mean pool
        reps = self._l2norm(reps)
        # return cls.detach().cpu().numpy()
        return reps.detach().cpu().numpy()

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> List[NDArray[np.float32]]:
        all_reps = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            reps = self._tokenize_and_encode(batch)
            all_reps.extend(reps) # concat batch
        return all_reps
    
    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: (B, L, d), attention_mask: (B, L)
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)
    
    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"])
    
class SimpEncoder(BaseEncoder):
    def __init__(
        self, 
        model_path: str, 
        tokenizer_path: str, 
        device: Union[str, torch.device], 
        max_length: int = 256, 
        batch_size: int = 1
    ):
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _tokenize_and_encode(self, texts: List[str]) -> NDArray[np.float32]:
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        reps = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])  # mean pool
        reps = self._l2norm(reps)
        return reps.detach().cpu().numpy()

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> List[NDArray[np.float32]]:
        all_reps = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            reps = self._tokenize_and_encode(batch)
            all_reps.extend(reps) # concat batch
        return all_reps
    
    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"])

class StructEncoder(BaseEncoder):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: Union[str, torch.device],
        max_length: int = 256,
        batch_size: int = 1,
    ):
        # Load wrapper config if present
        wrapper_config_path = os.path.join(model_path, "wrapper_config.json")
        wrapper_config = {}
        if os.path.exists(wrapper_config_path):
            with open(wrapper_config_path, "r", encoding="utf-8") as f:
                wrapper_config = json.load(f)

        self.pooling = str(wrapper_config.get("pooling", "cls"))
        self.l2_norm = bool(wrapper_config.get("l2_norm", True))

        # If wrapper config has max_nlq_length, prefer it unless caller explicitly changed max_length.
        wrapper_max_length = wrapper_config.get("max_nlq_length", None)
        if max_length == 128 and wrapper_max_length is not None:
            self.max_length = int(wrapper_max_length)
        else:
            self.max_length = max_length

        # Load model
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                add_pooling_layer=False,
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

        # Remove unused pooler if present
        if hasattr(self.model, "pooler") and self.model.pooler is not None:
            self.model.pooler = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )

        self.device = device
        self.batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _tokenize_and_encode(self, texts: List[str]) -> NDArray[np.float32]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state

        if self.pooling == "cls":
            reps = last_hidden_state[:, 0]
        elif self.pooling == "mean":
            reps = self._mean_pool(last_hidden_state, inputs["attention_mask"])
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        if self.l2_norm:
            reps = self._l2norm(reps)

        return reps.detach().cpu().numpy()

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> List[NDArray[np.float32]]:
        all_reps = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            reps = self._tokenize_and_encode(batch)
            all_reps.extend(reps)
        return all_reps

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)
    
    def count_tokens(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"])