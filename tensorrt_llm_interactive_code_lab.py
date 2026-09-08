#!/usr/bin/env python3
"""Interactive TensorRT-LLM code learning lab."""
from __future__ import annotations
import os, shutil, subprocess, sys, textwrap, time
from dataclasses import dataclass

MODEL_ID = os.getenv("TRTLLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

@dataclass(frozen=True)
class Lesson:
    number: int
    title: str
    definition: str
    code: str
    notes: str = ""


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def bar(ch="=", width=78):
    return ch * width


def pause(msg="\nPress Enter to continue..."):
    try: input(msg)
    except (EOFError, KeyboardInterrupt): pass


def wrap(text):
    print(textwrap.fill(text, width=78))


def show_code(code):
    print(bar("-")); print("CODE EXAMPLE"); print(bar("-"))
    for n, line in enumerate(code.strip().splitlines(), 1):
        print(f"{n:>3} | {line}")


def dependency_status():
    try:
        import tensorrt_llm
        return True, f"available ({getattr(tensorrt_llm, '__version__', 'version unknown')})"
    except Exception as exc:
        return False, f"unavailable: {type(exc).__name__}: {exc}"


def gpu_status():
    exe = shutil.which("nvidia-smi")
    if not exe: return "nvidia-smi not found"
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10)
        return result.stdout.strip() or "No GPU information returned"
    except Exception as exc:
        return f"Unable to query GPU: {exc}"

BASELINE_CODE = '''from tensorrt_llm import LLM, SamplingParams

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
)

sampling = SamplingParams(max_tokens=96, temperature=0.7, top_p=0.9)
outputs = llm.generate(
    ["Explain optimized LLM inference in four sentences."],
    sampling_params=sampling,
)

for output in outputs:
    print(output.outputs[0].text)
'''

FLASH_CODE = '''from tensorrt_llm import LLM, SamplingParams

# TensorRT-LLM selects optimized fused-attention kernels internally when the
# model, GPU, datatype, and sequence dimensions support them.
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
    max_input_len=2048,
    max_seq_len=4096,
)

outputs = llm.generate(
    ["Explain FlashAttention using a GPU-memory analogy."],
    sampling_params=SamplingParams(max_tokens=96),
)
print(outputs[0].outputs[0].text)
'''

KV_CODE = '''from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig

# Block reuse can reuse KV-cache pages for requests sharing the same prefix.
kv_cache = KvCacheConfig(
    enable_block_reuse=True,
    free_gpu_memory_fraction=0.85,
)

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
    kv_cache_config=kv_cache,
)

prefix = "You are an ML platform engineer. Answer clearly.\n"
prompts = [
    prefix + "What is paged KV caching?",
    prefix + "Why does KV-cache fragmentation matter?",
    prefix + "How does prefix reuse increase throughput?",
]

outputs = llm.generate(prompts, sampling_params=SamplingParams(max_tokens=96))
for output in outputs:
    print(output.outputs[0].text)
'''

FUSION_CODE = '''from tensorrt_llm import LLM, SamplingParams

# TensorRT-LLM performs supported graph and kernel fusion internally.
# Fewer launches and less intermediate memory traffic reduce latency.
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
)

prompts = [
    "How does kernel fusion reduce launch overhead?",
    "Why does reducing GPU memory traffic lower latency?",
]
outputs = llm.generate(prompts, sampling_params=SamplingParams(max_tokens=96))
for output in outputs:
    print(output.outputs[0].text)
'''

QUANT_CODE = '''from tensorrt_llm import LLM, SamplingParams

# Use a pre-quantized checkpoint supported by your TensorRT-LLM release.
QUANTIZED_MODEL = "YOUR_SUPPORTED_QUANTIZED_MODEL_OR_LOCAL_DIRECTORY"

llm = LLM(model=QUANTIZED_MODEL)
outputs = llm.generate(
    ["Compare FP16, FP8, INT8, and INT4 inference."],
    sampling_params=SamplingParams(max_tokens=96),
)
print(outputs[0].outputs[0].text)
'''

SPEC_CODE = '''from tensorrt_llm import LLM, SamplingParams

# Draft-target, EAGLE, Medusa, and n-gram speculative decoding use different
# release-specific configurations. This is the stable target-model baseline.
target_llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
)

outputs = target_llm.generate(
    ["Explain speculative decoding with a writer-and-editor analogy."],
    sampling_params=SamplingParams(max_tokens=128),
)
print(outputs[0].outputs[0].text)
'''

TP_CODE = '''from tensorrt_llm import LLM, SamplingParams

# Requires two visible compatible NVIDIA GPUs.
llm = LLM(
    model="YOUR_SUPPORTED_MODEL_OR_LOCAL_DIRECTORY",
    dtype="float16",
    tensor_parallel_size=2,
)

outputs = llm.generate(
    ["How does tensor parallelism divide matrix multiplication?"],
    sampling_params=SamplingParams(max_tokens=96),
)
print(outputs[0].outputs[0].text)
'''

PP_CODE = '''from tensorrt_llm import LLM, SamplingParams

# Requires two visible compatible NVIDIA GPUs.
llm = LLM(
    model="YOUR_SUPPORTED_MODEL_OR_LOCAL_DIRECTORY",
    dtype="float16",
    pipeline_parallel_size=2,
)

outputs = llm.generate(
    ["Explain pipeline parallelism using an assembly-line analogy."],
    sampling_params=SamplingParams(max_tokens=96),
)
print(outputs[0].outputs[0].text)
'''

BATCH_CODE = '''from tensorrt_llm import LLM, SamplingParams
import time

# Multiple prompts demonstrate batched generation. In production, the server
# scheduler continuously admits and completes differently sized requests.
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
)

prompts = [
    "Define time to first token.",
    "Define inter-token latency.",
    "Define inference throughput.",
    "Define continuous batching.",
]

started = time.perf_counter()
outputs = llm.generate(prompts, sampling_params=SamplingParams(max_tokens=72))
print(f"Elapsed: {time.perf_counter() - started:.2f} seconds")
for output in outputs:
    print(output.outputs[0].text)
'''

PLAYGROUND_CODE = '''from tensorrt_llm import LLM, SamplingParams

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype="float16",
)
sampling = SamplingParams(max_tokens=128, temperature=0.7, top_p=0.9)

print("Type /quit to leave.")
while True:
    prompt = input("Prompt > ").strip()
    if prompt.lower() in {"/quit", "/q", "quit", "q"}:
        break
    if not prompt:
        continue
    output = llm.generate([prompt], sampling_params=sampling)[0]
    print("\n" + output.outputs[0].text + "\n")
'''

LESSONS = [
    Lesson(1, "FlashAttention and fused attention",
      "FlashAttention reorganizes attention into tiles so intermediate work stays in fast on-chip GPU memory instead of repeatedly moving large tensors through global memory. TensorRT-LLM normally selects supported fused-attention kernels internally, so Python supplies model, datatype, and sequence settings rather than implementing the CUDA kernel manually.", FLASH_CODE),
    Lesson(2, "Paged KV cache and prefix reuse",
      "The KV cache stores attention keys and values from earlier tokens, preventing full-sequence recomputation for every new token. Paged allocation divides this memory into reusable blocks, reducing fragmentation, while prefix reuse can share blocks among requests beginning with identical prompt content.", KV_CODE,
      "KvCacheConfig fields and import paths can vary by TensorRT-LLM release."),
    Lesson(3, "Kernel fusion",
      "Kernel fusion combines compatible operations into fewer GPU kernels. This reduces launch overhead, synchronization, and intermediate GPU-memory traffic; TensorRT-LLM performs most fusion choices internally rather than requiring hand-written CUDA.", FUSION_CODE),
    Lesson(4, "Quantization",
      "Quantization represents weights and sometimes activations or KV-cache values with lower-precision formats such as FP8, INT8, or INT4. It can reduce VRAM use and memory bandwidth while increasing throughput, but model quality and hardware compatibility must be validated.", QUANT_CODE,
      "Replace the placeholder with a checkpoint supported by your installed release."),
    Lesson(5, "Speculative decoding",
      "Speculative decoding uses a cheaper draft mechanism to propose multiple likely next tokens and lets the target model verify them together. When enough proposals are accepted, generation advances by more than one token per expensive target-model pass, reducing latency.", SPEC_CODE,
      "Exact draft-target, EAGLE, Medusa, and n-gram setup is release-specific."),
    Lesson(6, "Tensor parallelism",
      "Tensor parallelism splits individual tensor operations across multiple GPUs, with the devices cooperating on the same Transformer layer. It helps when the model or workload needs more compute or memory than one GPU, although inter-GPU communication overhead matters.", TP_CODE,
      "Requires at least two compatible GPUs."),
    Lesson(7, "Pipeline parallelism",
      "Pipeline parallelism places different groups of Transformer layers on different GPUs and passes activations between stages. It can fit large models across devices, but good throughput requires enough concurrent work to limit idle pipeline bubbles.", PP_CODE,
      "Requires at least two compatible GPUs and suitable model partitioning."),
    Lesson(8, "In-flight or continuous batching",
      "In-flight batching lets the scheduler admit and complete requests dynamically instead of forcing every request to wait for a fixed batch. This improves GPU utilization when prompts and generated responses have different lengths and is especially valuable for production APIs.", BATCH_CODE),
]


def execute(code, title):
    available, status = dependency_status()
    print("\n" + bar("-")); print(f"RUNNING: {title}"); print(bar("-"))
    if not available:
        print("TensorRT-LLM is " + status)
        print("The code is viewable, but execution requires a compatible TensorRT-LLM environment.")
        pause(); return
    if input("This may download a model and allocate substantial GPU memory. Continue? [y/N] > ").strip().lower() not in {"y", "yes"}:
        print("Cancelled."); pause(); return
    try:
        result = subprocess.run([sys.executable, "-c", code], check=False)
        print(f"\nProcess exit code: {result.returncode}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"\nFailed: {type(exc).__name__}: {exc}")
    pause()


def playground():
    clear_screen(); print(bar()); print(" INTERACTIVE PROMPT PLAYGROUND"); print(bar())
    show_code(PLAYGROUND_CODE)
    if input("\n[R] Run  [B] Back > ").strip().lower() in {"r", "run"}:
        execute(PLAYGROUND_CODE, "Prompt playground")


def show_lesson(lesson, run_all=False):
    while True:
        clear_screen(); print(bar()); print(f" LESSON {lesson.number}: {lesson.title.upper()}"); print(bar())
        print("\nDEFINITION\n"); wrap(lesson.definition)
        if lesson.notes:
            print("\nNOTES\n"); wrap(lesson.notes)
        print(); show_code(lesson.code); print("\n" + bar("-"))
        print("[R] Run  [P] Playground  [N] Next  [M] Menu  [Q] Quit" if run_all else "[R] Run  [P] Playground  [B] Back  [Q] Quit")
        try: choice = input("\nChoice > ").strip().lower()
        except (EOFError, KeyboardInterrupt): return "quit"
        if choice in {"q", "quit"}: return "quit"
        if choice in {"b", "back", "m", "menu"}: return "menu"
        if run_all and choice in {"", "n", "next"}: return "next"
        if choice in {"p", "playground"}: playground()
        elif choice in {"r", "run"}: execute(lesson.code, lesson.title)
        else: print("Unknown choice."); time.sleep(.6)


def environment_check():
    clear_screen(); print(bar()); print(" ENVIRONMENT CHECK"); print(bar())
    available, status = dependency_status()
    print(f"\nPython          : {sys.version.split()[0]}")
    print(f"TensorRT-LLM    : {status}")
    print(f"NVIDIA GPU      : {gpu_status()}")
    print(f"TRTLLM_MODEL    : {MODEL_ID}")
    print(f"Execution       : {'enabled' if available else 'view-only'}")
    pause()


def menu():
    available, status = dependency_status()
    print(bar()); print(" TENSORRT-LLM INTERACTIVE CODE LEARNING LAB"); print(bar())
    print(f" SDK: {status}")
    print(f" GPU: {gpu_status()}")
    print(bar("-"))
    for lesson in LESSONS: print(f" {lesson.number}) {lesson.title}")
    print("\n 9) Run all lessons")
    print("10) Baseline code example")
    print("11) Interactive prompt playground")
    print("12) Environment check")
    print(" Q) Quit")
    print(bar("-"))


def main():
    while True:
        clear_screen(); menu()
        try: choice = input("\nSelection > ").strip().lower()
        except (EOFError, KeyboardInterrupt): return 0
        if choice in {"q", "quit", "exit"}: return 0
        if choice == "9":
            for lesson in LESSONS:
                result = show_lesson(lesson, True)
                if result == "quit": return 0
                if result == "menu": break
            continue
        if choice == "10":
            clear_screen(); print(bar()); print(" BASELINE CODE EXAMPLE"); print(bar()); show_code(BASELINE_CODE)
            if input("\n[R] Run  [B] Back > ").strip().lower() in {"r", "run"}: execute(BASELINE_CODE, "Baseline")
            continue
        if choice == "11": playground(); continue
        if choice == "12": environment_check(); continue
        try: number = int(choice)
        except ValueError: print("Choose 1-12 or Q."); time.sleep(.7); continue
        lesson = next((x for x in LESSONS if x.number == number), None)
        if lesson and show_lesson(lesson) == "quit": return 0

if __name__ == "__main__":
    raise SystemExit(main())
