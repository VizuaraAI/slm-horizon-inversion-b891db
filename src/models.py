"""Model wrappers: a local small language model, and a simulator for testing.

`LocalChatModel` is the real thing: any instruction-tuned checkpoint from the
Hugging Face hub, loaded once and sampled many times. Transformers is imported
lazily so that the metrics and horizon code stay usable on a laptop with no
GPU and no model downloaded.

`SimulatedStepper` is not a stand-in for a model. It is a measurement tool:
it emits correct or corrupted step answers at a per-step accuracy you set, with
a tunable amount of error correlation. Use it to check that your estimator
recovers the per-step accuracy and the effective horizon you put in, before you
trust the same estimator on a real checkpoint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful step executor. You are given the running state of a "
    "multi-step task and one instruction. Produce only the result of that "
    "single step. Do not solve later steps. Do not explain."
)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 48
    temperature: float = 0.0
    top_p: float = 0.95
    num_samples: int = 1


class LocalChatModel:
    """Thin wrapper over a Hugging Face causal language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        dtype: str = "bfloat16",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
        )
        self.model.eval()
        self._torch = torch

    def generate(self, prompt: str, cfg: GenerationConfig) -> List[str]:
        """Return `cfg.num_samples` completions for one prompt."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        do_sample = cfg.temperature > 0.0
        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=do_sample,
                temperature=cfg.temperature if do_sample else None,
                top_p=cfg.top_p if do_sample else None,
                num_return_sequences=cfg.num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            for seq in output
        ]

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])


@dataclass
class SimulatedStepper:
    """Emits step answers at a controlled per-step accuracy.

    p_base        accuracy of a step while the rollout is still on-track
    p_after_error probability of producing the gold value once an earlier
                  step was wrong, which is the recovery probability. Set it
                  to 0 for a pure chain where one error is fatal, in which
                  case end-to-end success is exactly p_base ** N and the
                  fitted alpha should come out at 1.
    decay         per-step drop in p_base along the chain, which models
                  self-conditioning. Any positive value pushes alpha above 1.
    seed          reproducibility
    """

    p_base: float = 0.90
    p_after_error: float = 0.0
    decay: float = 0.0
    seed: int = 0
    model_name: str = "simulated"
    _rng: random.Random = field(default=None, repr=False)
    _step_index: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        """Called at the start of every rollout so decay restarts at step 1."""
        self._step_index = 0

    def step(self, gold: str, saw_error: bool) -> str:
        index = self._step_index
        self._step_index += 1
        if saw_error:
            p = self.p_after_error
        else:
            p = max(0.0, min(1.0, self.p_base - self.decay * index))
        if self._rng.random() < p:
            return gold
        return self._corrupt(gold)

    def _corrupt(self, gold: str) -> str:
        try:
            value = float(gold)
        except ValueError:
            return gold + " (incorrect)"
        offset = self._rng.choice([-3, -2, -1, 1, 2, 3])
        if value.is_integer():
            return str(int(value) + offset)
        return f"{value + offset:.2f}"

    def generate(self, prompt: str, cfg: GenerationConfig) -> List[str]:
        raise NotImplementedError(
            "SimulatedStepper is driven through .step(); it does not read prompts."
        )


def load_model(spec: str, **kwargs):
    """Load either a real checkpoint or the simulator.

    spec == "simulated"                -> SimulatedStepper
    spec == "simulated:0.92:0.3"       -> p_base 0.92, recovery 0.3
    spec == "simulated:0.92:0.0:0.01"  -> and a per-step decay of 0.01
    anything else                      -> LocalChatModel(spec)
    """
    if spec.startswith("simulated"):
        parts = spec.split(":")
        p_base = float(parts[1]) if len(parts) > 1 else 0.90
        p_after = float(parts[2]) if len(parts) > 2 else 0.0
        decay = float(parts[3]) if len(parts) > 3 else 0.0
        return SimulatedStepper(
            p_base=p_base, p_after_error=p_after, decay=decay, **kwargs
        )
    return LocalChatModel(spec, **kwargs)


SUGGESTED_CHECKPOINTS: Sequence[str] = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)


def suggested_teacher() -> Optional[str]:
    """A larger model to distil step supervision from, if you have the memory."""
    return "Qwen/Qwen2.5-32B-Instruct"
