"""Model execution boundary for the serving pipeline.

This module defines the common Backend interface used by the scheduler pipeline.
The rest of the system calls run_batch(...) and receives normalized
BackendResult objects, regardless of whether the implementation is a local mock
or a real vLLM/Qwen2-VL backend.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from pathlib import Path
from typing import Any

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
            start_time = time.time()
            first_token_time = start_time
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
    """Optional vLLM backend for running a real multimodal model."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2-VL-7B-Instruct",
        max_tokens: int = 128,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Use MockBackend for local smoke tests, "
                "or install vLLM in the model-serving environment."
            ) from exc

        self._llm = LLM(model=model, trust_remote_code=trust_remote_code)
        self._sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    def generate(self, request: MMRequest) -> BackendResult:
        """Model-wrapper function for a single request."""
        return self.run_request(request)

    def run_batch(self, batch: Batch) -> list[BackendResult]:
        """Run a batch through vLLM and normalize outputs."""
        inputs = [self._to_vllm_input(request) for request in batch.requests]
        start_time = time.time()
        outputs = self._llm.generate(inputs, self._sampling_params)
        completion_time = time.time()

        results: list[BackendResult] = []
        for request, output in zip(batch.requests, outputs, strict=True):
            generated_text = output.outputs[0].text if output.outputs else ""
            token_ids = getattr(output.outputs[0], "token_ids", None) if output.outputs else None
            output_token_count = len(token_ids) if token_ids is not None else len(generated_text.split())
            results.append(
                BackendResult(
                    request_id=request.request_id,
                    generated_text=generated_text,
                    output_token_count=output_token_count,
                    first_token_time=start_time,
                    completion_time=completion_time,
                    raw={
                        "vllm_request_id": getattr(output, "request_id", None),
                        "finish_reason": getattr(output.outputs[0], "finish_reason", None)
                        if output.outputs
                        else None,
                    },
                )
            )
        return results

    def _to_vllm_input(self, request: MMRequest) -> str | dict[str, Any]:
        """Format text-only or image+text requests for vLLM."""
        if not request.image_path:
            return request.prompt

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for image requests with VLLMBackend.") from exc

        image_path = Path(request.image_path)
        with Image.open(image_path) as image:
            loaded_image = image.convert("RGB")

        return {
            "prompt": f"<image>\n{request.prompt}",
            "multi_modal_data": {"image": loaded_image},
        }
