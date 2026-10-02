from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class Text2CypherMethod:
    def __init__(self):
        pass

class QwenvLLM(Text2CypherMethod):
    def __init__(
        self, 
        model_path: str = "Qwen/Qwen2.5-72B-Instruct",
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 2,
        max_model_length: int = 4096,
        vllm_memory_utilization: float = 0.9,
        swap_space: int = 8,
        max_num_seqs: int = 1,
        max_num_batched_tokens: int = 4096,
        sampling_max_tokens: int = 1024,
        sampling_temperature: float = 0,
        sampling_n: int = 1,
    ):

    def __init__(self, model_path: str = "Qwen/Qwen2.5-72B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature = sampling_temperature, 
            max_tokens = sampling_max_tokens,
            n = sampling_n
        )

        self.llm = LLM(
            model = model_path,
            tokenizer = model_path,
            dtype = dtype,
            tensor_parallel_size = tensor_parallel_size,
            max_model_len = max_model_length,
            gpu_memory_utilization = vllm_memory_utilization,
            max_num_seqs = max_num_seqs,
            max_num_batched_tokens = max_num_batched_tokens,
            enforce_eager = True,
            disable_custom_all_reduce = False,
            trust_remote_code = True,
        )

        self._recent_chat_prompt = ""
        self._recent_outputs = []

    def generate(self, input_prompt: str):
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": input_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

        outputs = self.llm.generate([chat_prompt], self.sampling_params)

        self._recent_chat_prompt = chat_prompt
        self._recent_outputs = outputs
        return outputs

    def get_recent_input_output_token_length(self):
        input_token_length = len(
            self.tokenizer.encode(
                self._recent_chat_prompt,
                add_special_tokens=False,
            )
        )

        llm_output_token_length = 0
        for output in self._recent_outputs:
            if output.outputs:
                response_text = output.outputs[0].text.strip()
                token_ids = getattr(output.outputs[0], "token_ids", None)

                if token_ids is not None:
                    llm_output_token_length = len(token_ids)
                else:
                    llm_output_token_length = len(
                        self.tokenizer.encode(
                            response_text,
                            add_special_tokens=False,
                        )
                    )
                break

        return input_token_length, llm_output_token_length