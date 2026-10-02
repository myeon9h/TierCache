import torch
from vllm import LLM, SamplingParams
from typing import Tuple, Any
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast

class Text2SQLMethod:
    def __init__(self):
        pass

class OmniSQLvLLM(Text2SQLMethod):
    def __init__(
        self, 
        model_path: str = "seeklhy/OmniSQL-7B",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        max_model_length: int = 8192,
        vllm_memory_utilization: float = 0.9,
        swap_space: int = 8,
        sampling_max_tokens: int = 2048,
        sampling_temperature: float = 0,
        sampling_n: int = 1,
    ):
        
        # Init Model & Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.sampling_params = SamplingParams(
            temperature = sampling_temperature, 
            max_tokens = sampling_max_tokens,
            n = sampling_n
        )

        self.llm = LLM(
            model = model_path,
            dtype = dtype, 
            tensor_parallel_size = tensor_parallel_size,
            max_model_len = max_model_length,
            gpu_memory_utilization = vllm_memory_utilization,
            swap_space = swap_space,
            enforce_eager = True,
            disable_custom_all_reduce = True,
            trust_remote_code = True
        )

        self._recent_chat_prompt = ""
        self._recent_outputs = []

    def generate(self, input_prompt: str) -> Any:
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": input_prompt}],
            add_generation_prompt = True, tokenize = False
        )
        outputs = self.llm.generate([chat_prompt], self.sampling_params)

        self._recent_chat_prompt = chat_prompt
        self._recent_outputs = outputs
        return outputs

    def get_recent_input_output_token_length(self) -> Tuple[int, int]:
        input_token_length = len(self.tokenizer.encode(self._recent_chat_prompt, add_special_tokens=False))

        for output in self._recent_outputs:
            if output.outputs:
                response_text = output.outputs[0].text.strip()
                token_ids = getattr(output.outputs[0], "token_ids", None)
                if token_ids is not None:
                    llm_output_token_length = len(token_ids)
                else:
                    llm_output_token_length = len(
                        self.tokenizer.encode(response_text, add_special_tokens=False)
                    )
                break

        return input_token_length, llm_output_token_length 

class T5(Text2SQLMethod):
    def __init__(
        self,
        model_path: str,
        max_input_length: int = 3072,
        max_output_length: int = 384,
        device: str = None,
    ):
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    @torch.no_grad()
    def generate(self, input_prompt: str) -> str:
        enc = self.tokenizer(
            input_prompt,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        ).to(self.device)

        out_ids = self.model.generate(
            **enc,
            max_new_tokens=self.max_output_length,
            num_beams=4,
            length_penalty=0.6,
            early_stopping=True,
        )
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return text.strip()


        