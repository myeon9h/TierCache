import re
import torch
from typing import List, Tuple, Union, Optional
from transformers import T5Tokenizer, T5ForConditionalGeneration
from ..utils.special_tokens import SPECIAL_SEP, SPECIAL_MASK

class BaseSlotFiller:
    def __init__(self):
        pass

class T5LiteralFiller(BaseSlotFiller):
    def __init__(
        self,
        model_path: str,
        device: Union[str, torch.device],
        max_input_length: int = 384,
        max_output_length: int = 128,
    ):
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            use_safetensors=True,
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.device = device
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self._current_input_token_length = 0
        self._current_output_token_length = 0

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def literal_filling(self, nlq: str, sq_template: str) -> str:
        num_masked_literals = sq_template.count(SPECIAL_MASK)

        if num_masked_literals == 0:
            return sq_template

        prompt = self._build_prompt(nlq, sq_template, num_masked_literals)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_output_length,
        )
        pred = self.tokenizer.decode(gen[0].cpu(), skip_special_tokens=False)
        literals = self._preprocess_pred(
            pred,
            remove_target_tokens=[
                self.tokenizer.eos_token,
                self.tokenizer.pad_token,
                self.tokenizer.unk_token,
            ],
        )

        self._current_input_token_length = int(inputs["attention_mask"][0].sum().item())
        self._current_output_token_length = int(gen[0].shape[0])

        if len(literals) != num_masked_literals:
            return f"{sq_template} (Filling Error: expected {num_masked_literals} literals, got {len(literals)})"

        return self._fill_template(sq_template, literals)

    def _build_prompt(self, nlq: str, sq_template: str, num_slots: int) -> str:
        slot_tokens = " ".join([f"<extra_id_{i}>" for i in range(num_slots)])
        return (
            f"fill literals in template {SPECIAL_SEP} "
            f"question: {nlq} {SPECIAL_SEP} "
            f"template: {sq_template} {SPECIAL_SEP} "
            f"slots: {slot_tokens}"
        )

    def _preprocess_pred(
        self,
        pred: str,
        remove_target_tokens: Optional[List[str]] = None,
    ) -> List[str]:
        if remove_target_tokens is None:
            remove_target_tokens = ["UNK"]

        for t in remove_target_tokens:
            if t is not None:
                pred = pred.replace(t, "")
        pred = pred.strip()

        matches = re.findall(
            r"<extra_id_(\d+)>\s*(.*?)(?=\s*<extra_id_\d+>|$)",
            pred,
            flags=re.DOTALL,
        )

        matches = sorted(
            [(int(idx), lit.strip()) for idx, lit in matches],
            key=lambda x: x[0],
        )

        literals = [lit for _, lit in matches if len(lit) > 0]
        return literals

    # Replace each SPECIAL_MASK with the corresponding literal
    def _fill_template(self, sq_templ: str, literals: List[str]) -> str:
        num_literal_tokens = sq_templ.count(SPECIAL_MASK)

        if num_literal_tokens != len(literals):
            return f"{sq_templ} (Filling Error: expected {num_literal_tokens} literals, got {len(literals)})"

        parts = sq_templ.split(SPECIAL_MASK)
        final_sq_chunks: List[str] = []

        for i in range(len(parts) - 1):
            final_sq_chunks.append(parts[i])
            final_sq_chunks.append(literals[i].strip())

        final_sq_chunks.append(parts[-1])
        return "".join(final_sq_chunks)

    def get_recent_input_output_token_lengths(self) -> Tuple[int, int]:
        return self._current_input_token_length, self._current_output_token_length
    
    # For debugging: return both filled template and predicted literals
    @torch.inference_mode()
    def literal_filling_and_get_pred_literals(
        self,
        nlq: str,
        sq_template: str,
    ) -> Tuple[str, List[str]]:
        num_masked_literals = sq_template.count(SPECIAL_MASK)

        if num_masked_literals == 0:
            return sq_template, []

        prompt = self._build_prompt(nlq, sq_template, num_masked_literals)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        gen = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_output_length,
        )
        pred = self.tokenizer.decode(gen[0].cpu(), skip_special_tokens=False)
        literals = self._preprocess_pred(
            pred,
            remove_target_tokens=[
                self.tokenizer.eos_token,
                self.tokenizer.pad_token,
                self.tokenizer.unk_token,
            ],
        )

        if len(literals) != num_masked_literals:
            return f"{sq_template} (Filling Error: expected {num_masked_literals} literals, got {len(literals)})", literals

        return self._fill_template(sq_template, literals), literals