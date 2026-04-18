"""Model execution boundary for the serving pipeline.

This module defines the common Backend interface used by the scheduler pipeline.
The rest of the system calls run_batch(...) and receives normalized
BackendResult objects, regardless of whether the implementation is a local mock
or a real vLLM/Qwen2-VL backend.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from mmserve_skeleton.models import BackendResult, Batch, MMRequest


class Backend(ABC):
    """Abstract interface every execution backend must implement."""

    @abstractmethod
    def run_batch(self, batch: Batch) -> list[BackendResult]:
        """Run one scheduled batch and return one result per request."""
        raise NotImplementedError

    def run_request(self, request: MMRequest) -> BackendResult:
        """Convenience wrapper for callers that want to execute one request."""
        batch = Batch(batch_id="single", requests=[request], created_time=time.time())
        return self.run_batch(batch)[0]


class MockBackend(Backend):
    """Fast fake backend for testing queueing, logging, and analysis locally."""

    def __init__(self, sleep_seconds: float = 0.01) -> None:
        self.sleep_seconds = sleep_seconds

    def run_batch(self, batch: Batch) -> list[BackendResult]:
        results: list[BackendResult] = []
        for request in batch.requests:
            first_token_time = time.time()
            time.sleep(self.sleep_seconds)
            generated_text = f"[mock] Answer for request {request.request_id}: {request.prompt[:80]}"
            completion_time = time.time()
            results.append(
                BackendResult(
                    request_id=request.request_id,
                    generated_text=generated_text,
                    output_token_count=len(generated_text.split()),
                    first_token_time=first_token_time,
                    completion_time=completion_time,
                )
            )
        return results


class VLLMBackend(Backend):
    """
    Placeholder for the real vLLM/Qwen2-VL backend implementation.
    """

    # def __init__(
    #     self,
    #     model: str = "Qwen/Qwen2-VL-7B-Instruct",
    #     max_tokens: int = 128,
    #     temperature: float = 0.0,
    # ) -> None:
    #     try:
    #         from vllm import LLM, SamplingParams
    #     except ImportError as exc:
    #         raise RuntimeError(
    #             "vLLM is not installed. Use --backend mock for local smoke tests "
    #             "or install vLLM in the model-serving environment."
    #         ) from exc

    #     self._llm = LLM(model=model)
    #     self._sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    # def run_batch(self, batch: Batch) -> list[BackendResult]:
    #     prompts = [self._format_prompt(request) for request in batch.requests]
    #     dispatch_time = time.time()
    #     outputs = self._llm.generate(prompts, self._sampling_params)
    #     completion_time = time.time()

    #     results: list[BackendResult] = []
    #     for request, output in zip(batch.requests, outputs, strict=True):
    #         generated_text = output.outputs[0].text if output.outputs else ""
    #         results.append(
    #             BackendResult(
    #                 request_id=request.request_id,
    #                 generated_text=generated_text,
    #                 output_token_count=len(generated_text.split()),
    #                 first_token_time=dispatch_time,
    #                 completion_time=completion_time,
    #                 raw={"vllm_request_id": getattr(output, "request_id", None)},
    #             )
    #         )
    #     return results

    # def _format_prompt(self, request: MMRequest) -> str:
    #     if request.image_path:
    #         return f"<image>{request.image_path}</image>\n{request.prompt}"
    #     return request.prompt
