from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

from src.refusal_judge_prompt import (
    extract_refusal_score,
    get_judge_system_prompt,
    get_judge_user_prompt,
)
from src.utils import delete_llm, encode_conversation

_SYSTEM_PROMPT = get_judge_system_prompt()


class LLMJudge:
    """
    A judge model for evaluating whether LLM responses constitute refusals.

    Uses a specified LLM to analyze question-answer pairs and classify whether
    the answer represents a refusal to answer or a genuine attempt to address
    the question.

    The instruction block (~6200 tokens) is placed in a system message so that
    vLLM's prefix caching computes the KV cache once and reuses it for every
    request, avoiding repeated prefill of the shared instructions.
    """

    def __init__(
        self,
        model_name: str,
        max_model_len: int,
        gpu_memory_utilization: float = 0.95,
        tensor_parallel_size: int = torch.cuda.device_count(),
        kv_cache_dtype: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=False,
            kv_cache_dtype=kv_cache_dtype,
            max_model_len=max_model_len,
        )

        self.extract_score = extract_refusal_score

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def judge(
        self,
        questions_answers: List[Tuple[str, str]],
        num_return_sequences: int = 1,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_new_tokens: int = 8196,
        thinking_string: Optional[str] = None,
    ) -> List[Dict[str, Union[str, float, List[str]]]]:
        """
        Evaluate conversations to determine if responses constitute refusals.

        Takes a list of conversations and uses the LLM to judge whether each
        response represents a refusal to answer or a genuine attempt to address
        the question. Returns classification scores and judge outputs.

        Args:
            questions_answers: List of (question, answer) tuples.
            num_return_sequences: Number of sampled outputs per input.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling parameter.
            top_k: Top-k sampling parameter.
            max_new_tokens: Maximum tokens to generate.
            thinking_string: String to split on for chain-of-thought reasoning.

        Returns:
            List of dicts with 'label' (float) and 'judge_outputs' (List[str]).
        """
        if not questions_answers:
            return []

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
        )

        conversations: List[List[Dict[str, str]]] = [
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": get_judge_user_prompt(question, answer)},
            ]
            for question, answer in questions_answers
        ]

        batch_messages: List[TokensPrompt] = encode_conversation(
            conversations=conversations,
            tokenizer=self.tokenizer,
            max_model_len=self.max_model_len,
            max_new_tokens=max_new_tokens,
        )

        if thinking_string == "":
            thinking_string = None

        outputs = self.llm.generate(
            prompts=batch_messages,
            sampling_params=sampling_params,
            use_tqdm=None,
        )

        results: List[Dict[str, Union[str, float, List[str]]]] = []
        for result in outputs:
            example_output: Dict[str, Union[str, float, List[str]]] = {
                "label": 0.0,
                "judge_outputs": [],
            }
            valid = 0
            total_score = 0.0
            for output in result.outputs:
                text = output.text
                if thinking_string is not None:
                    text_split = text.split(thinking_string)[-1]
                else:
                    text_split = text
                score = self.extract_score(text_split)
                example_output["judge_outputs"].append(text)
                if score == 0.0:
                    continue
                valid += 1
                total_score += score

            if valid == 0:
                example_output["label"] = 0.0
            else:
                example_output["label"] = total_score / valid
            results.append(example_output)
        return results

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, "llm"):
            delete_llm(self.llm)
            self.llm = None
