from typing import (
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

from src.utils import delete_llm, encode_conversation


def geom_mean_prob(logs: List[float]) -> float:
    """Compute a length-normalized probability from token log-probabilities.

    Args:
        logs: Sequence of log-probabilities (log p_t) for each token.

    Returns:
        The geometric mean probability, computed as exp(mean(log p_t)).
        Returns -1.0 when no tokens are provided (sentinel for "no segment").
    """
    if logs is None or len(logs) == 0:
        return -1.0  # sentinel for "no segment"
    avg_log = torch.tensor(logs, dtype=torch.float32).mean()
    try:
        return float(torch.exp(avg_log).item())
    except (OverflowError, RuntimeError):
        # If exp(avg_log) overflows, return 1.0 (effectively 100% probability)
        return 1.0


class TokenCandidate(Protocol):
    """Protocol for a token candidate carrying ranking and logprob metadata."""

    rank: int
    logprob: float
    decoded_token: str


def parse_log_progs(
    logprobs: Sequence[Mapping[str, TokenCandidate]],
    thinking_string: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Parse per-token log-probabilities into thinking/answer segments.

    Args:
        logprobs: A sequence where each element corresponds to one position in
            the generated sequence and is a mapping from decoded token strings
            to a candidate object with attributes `rank`, `logprob`, and
            `decoded_token`. The highest-rank candidate is assumed to be the
            chosen token for that position.
        thinking_string: Optional special token/string that delimits the end of
            the "thinking" segment and the beginning of the answer segment. If
            not provided, the entire sequence is treated as the answer segment.

    Returns:
        A tuple of three floats:
        - thinking_prob: Geometric-mean probability of the thinking segment;
          0.0 if `thinking_string` is None or no thinking segment present.
        - answer_prob: Geometric-mean probability of the answer segment.
        - cum: Combined geometric-mean probability across both segments,
          removing length bias by averaging logs across all tokens considered.
    """
    # Collect logprobs for segments
    logprobs_answer: List[float] = []
    logprobs_think: List[float] = []
    current: List[float]
    if thinking_string is None:
        current = logprobs_answer
    else:
        current = logprobs_think

    # Track accumulated decoded tokens to match multi-token thinking_string
    accumulated_text = ""

    for output in logprobs:
        values = list(output.values())
        if not values:
            # Skip positions with no logprob data
            continue
        chosen = max(values, key=lambda x: x.rank)

        if thinking_string is not None:
            accumulated_text += chosen.decoded_token
            # Check if the thinking_string appears at the end of accumulated text
            # Switch to answer segment BEFORE appending logprob for delimiter tokens
            if accumulated_text.endswith(thinking_string):
                current = logprobs_answer

        current.append(chosen.logprob)

    if thinking_string is None:
        answer_prob = geom_mean_prob(logprobs_answer)
        thinking_prob = 0.0
        cum = answer_prob  # combined is just answer segment
    else:
        thinking_prob = geom_mean_prob(logprobs_think)
        answer_prob = geom_mean_prob(logprobs_answer)
        # Combine by averaging logs across both segments (removes length bias overall)
        both = (logprobs_think or []) + (logprobs_answer or [])
        cum = geom_mean_prob(both)

    return thinking_prob, answer_prob, cum


class GenerateAnswers:
    """Wrapper for generating answers and extracting log-probability metrics."""

    def __init__(
        self,
        model_name: str,
        max_model_len: int,
        gpu_memory_utilization: float = 0.95,
        tensor_parallel_size: int = torch.cuda.device_count(),
        enforce_eager: bool = False,
        kv_cache_dtype: str = "auto",
    ):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            kv_cache_dtype=kv_cache_dtype,
            max_model_len=max_model_len,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_answers(
        self,
        questions: List[str],
        max_new_tokens: int,
        num_return_sequences: int,
        thinking_string: Optional[str] = None,
        strip_prompt: bool = False,
        repetition_penalty: float = 1.0,
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """Generate model answers and compute segment-wise probabilities.

        Args:
            questions: Batch of questions.
            max_new_tokens: Maximum number of new tokens to generate.
            num_return_sequences: Number of sampled outputs per input conversation.
            thinking_string: Optional special token/string that separates the
                thinking segment from the answer segment for probability parsing.
            strip_prompt: Whether to strip the prompt to remove whitespaces and newlines.
            repetition_penalty: Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
        Returns:
            A nested list of outputs. For each input conversation, returns a list
            of dictionaries (one per returned sequence) containing:
            - "text": The generated text (str).
            - "thinking_prob": Geometric-mean probability of thinking segment (float).
            - "answer_prob": Geometric-mean probability of answer segment (float).
            - "cum": Combined geometric-mean probability across segments (float).
        """
        if not questions:
            return []

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            top_p=1.0,
            top_k=-1,
            temperature=1.0,
            logprobs=1,
            n=num_return_sequences,
            repetition_penalty=repetition_penalty,
        )
        conversations: List[List[Dict[str, str]]] = [
            [{"role": "user", "content": question.strip()}] for question in questions
        ]

        batch_messages: List[TokensPrompt] = encode_conversation(
            conversations=conversations,
            tokenizer=self.tokenizer,
            max_model_len=self.max_model_len,
            max_new_tokens=max_new_tokens,
            strip_prompt=strip_prompt,
        )

        results = self.llm.generate(
            prompts=batch_messages,
            sampling_params=sampling_params,
            use_tqdm=None,
        )

        outputs: List[List[Dict[str, Union[str, float]]]] = []

        for result in results:
            example_output: List[Dict[str, Union[str, float]]] = []
            for output in result.outputs:
                text = output.text
                thinking_prob, answer_prob, cum = parse_log_progs(
                    output.logprobs, thinking_string
                )
                example_output.append(
                    {
                        "text": text,
                        "thinking_prob": thinking_prob,
                        "answer_prob": answer_prob,
                        "cum": cum,
                    }
                )

            outputs.append(example_output)

        return outputs

    def __del__(self):
        self.close()

    def close(self):
        """Delete the LLM instance and free the GPU memory."""
        if hasattr(self, "llm"):
            delete_llm(self.llm)
            self.llm = None
