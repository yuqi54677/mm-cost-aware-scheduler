"""Model execution boundary for the serving pipeline.

This module defines the common Backend interface used by the scheduler pipeline.
The rest of the system calls run_batch(...) and receives normalized
BackendResult objects, regardless of whether the implementation is a local mock
or a real vLLM/Qwen2-VL backend.

VLLMBackend uses AsyncLLMEngine with token streaming so that first_token_time
reflects the actual arrival of the first generated token, not batch start time.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .models import BackendResult, Batch, MMRequest


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


def _run_coroutine(coro: Any) -> Any:
    """Run a coroutine whether or not an event loop is already running.

    In a plain script, asyncio.run() works fine. When called from within a
    running loop (e.g. Jupyter), we offload to a thread to avoid the
    'This event loop is already running' error.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


class VLLMBackend(Backend):
    """vLLM backend with async streaming for accurate TTFT measurement.

    Uses AsyncLLMEngine so each request's first_token_time is captured the
    moment the first token is yielded from the model, not at batch-start time.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2-VL-2B-Instruct",
        max_tokens: int = 128,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
    ) -> None:
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Use MockBackend for local smoke tests, "
                "or install vLLM in the model-serving environment."
            ) from exc

        engine_args = AsyncEngineArgs(model=model, trust_remote_code=trust_remote_code)
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)

    def generate(self, request: MMRequest) -> BackendResult:
        """Model-wrapper function for a single request."""
        return self.run_request(request)

    def run_batch(self, batch: Batch) -> list[BackendResult]:
        """Run all batch requests concurrently and return results in order."""
        return _run_coroutine(self._run_batch_async(batch))

    async def _run_batch_async(self, batch: Batch) -> list[BackendResult]:
        tasks = [self._run_single_async(req) for req in batch.requests]
        return list(await asyncio.gather(*tasks))

    async def _run_single_async(self, request: MMRequest) -> BackendResult:
        """Stream one request through vLLM, capturing TTFT on the first token."""
        prompt = await self._to_vllm_input(request)
        first_token_time: float | None = None
        final_output = None

        async for output in self._engine.generate(
            prompt, self._sampling_params, request.request_id
        ):
            if (
                first_token_time is None
                and output.outputs
                and output.outputs[0].token_ids
            ):
                first_token_time = time.time()
            final_output = output

        completion_time = time.time()

        if final_output is None or not final_output.outputs:
            return BackendResult(
                request_id=request.request_id,
                generated_text="",
                output_token_count=0,
                first_token_time=first_token_time,
                completion_time=completion_time,
            )

        out = final_output.outputs[0]
        generated_text = out.text
        token_ids = getattr(out, "token_ids", None)
        output_token_count = len(token_ids) if token_ids is not None else len(generated_text.split())

        return BackendResult(
            request_id=request.request_id,
            generated_text=generated_text,
            output_token_count=output_token_count,
            first_token_time=first_token_time,
            completion_time=completion_time,
            raw={
                "vllm_request_id": getattr(final_output, "request_id", None),
                "finish_reason": getattr(out, "finish_reason", None),
            },
        )

    async def _to_vllm_input(self, request: MMRequest) -> dict[str, Any]:
        """Format requests through the chat template so vLLM's Renderer sees them as non-raw."""
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = self._engine.get_tokenizer()

        if not request.image_path:
            messages = [{"role": "user", "content": request.prompt}]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return {"prompt": text}

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for image requests with VLLMBackend.") from exc

        image_path = Path(request.image_path)
        with Image.open(image_path) as image:
            loaded_image = image.convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": request.prompt},
        ]}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": text, "multi_modal_data": {"image": loaded_image}}
