# vllm-ios

**vLLM-style continuous batching for iPhone. Native Swift on MLX, no Python.**

vllm-ios is an embeddable inference engine focused on one thing: the fastest
multi-agent LLM serving on iOS. If your app runs several model calls
concurrently — agent fleets, parallel RAG extraction, fan-out research — this
engine batches them through one set of weights so every forward pass and every
byte of memory bandwidth does maximum work.

On an iPhone, that matters more than anywhere else: you get roughly **15–20
seconds of full-speed GPU per minute** before thermal throttling takes over.
Whatever inference you're doing needs to be over as fast as possible.
Background and origin story: [Continuous Batching on an
iPhone](https://jonready.com/blog/posts/continuous-batching-on-an-iphone.html).

> Unaffiliated with the [vLLM](https://github.com/vllm-project/vllm) project.
> Architecturally inspired by vLLM and
> [vllm-mlx](https://arxiv.org/abs/2601.19139); shares no code with either.

## Measured on an iPhone 16 Pro

Qwen3.5-0.8B, 4-bit, greedy decoding, thermally controlled runs:

| Batch size | Per-stream | Aggregate decode | Speedup |
|-----------:|-----------:|-----------------:|--------:|
| 1          | 103 tok/s  | 103 tok/s        | 1.0x    |
| 2          |  84 tok/s  | 168 tok/s        | 1.6x    |
| 8          |  25 tok/s  | 199 tok/s        | 1.9x    |

At 8-bit, against llama.cpp with functionally identical weights (Q8_0 vs MLX
8-bit), level for level:

| Concurrency | llama.cpp Q8_0 | vllm-ios (MLX 8-bit) |
|------------:|---------------:|---------------------:|
| 1           | 45 tok/s       | 51 tok/s             |
| 2           | 70 tok/s       | 95 tok/s             |
| 4           | 87 tok/s       | 161 tok/s            |
| 8           | 90 tok/s       | 169 tok/s            |

End to end: **16 research requests (~17k prompt tokens, 2k generated tokens of
structured JSON) in 25 seconds**, inside one thermal budget, with sub-second
admission of new requests into a running batch (best measured TTFT under load:
0.94s).

Scheduling overhead is measurably zero: per-stream throughput inside the full
engine matches a hand-rolled static batch exactly.

## How it works

The engine is ~300 lines on top of
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm), using stock MLX
kernels and public API only. Three ideas:

1. **Uniform-offset continuous batching.** Every sequence in the running batch
   always shares the same KV offset, so the model's stock causal-mask path
   works unchanged — no ragged attention masks, no model forks. A request
   arriving mid-flight is left-padded with filler tokens to the group's
   current offset, prefilled solo, then spliced in. For workloads with
   similar-length prompts the padding waste is ~1%.
2. **Cache surgery.** Join and exit happen at token boundaries by
   concatenating or index-selecting each layer's KV tensors along the batch
   axis — via `KVCache.state` accessors and `ArraysCache.extend/filter`, so
   it works across cache types, including the recurrent caches of hybrid
   models like Qwen3.5 (GatedDeltaNet).
3. **Chained greedy decode.** With greedy sampling the token-feedback loop
   never needs the CPU: several decode steps are built into one lazy graph
   (argmax feeds the next step on-GPU) and evaluated with a single sync per
   chunk.

Prefill runs in chunks that evaluate only cache state, so lm_head output for
prompt positions is never computed — without this, batched prefill of 8×1k-token
prompts materializes a ~2.6 GB logits tensor and iOS jetsam kills the app.

## Usage

```swift
import VLLMiOS
import MLXLMCommon

// context: a ModelContext from mlx-swift-lm (any LLM it can load)
let engine = VLLMEngine(
    model: context.model,
    padToken: 198,          // a harmless filler token, e.g. "\n"
    maxBatch: 4             // 2–4 is the sweet spot on A-series
)

let requests = prompts.enumerated().map { i, tokens in
    EngineRequest(id: i, promptTokens: tokens, maxTokens: 128,
                  arrivalTime: CFAbsoluteTimeGetCurrent())
}
let report = try engine.run(requests: requests)
for result in report.results {
    print(result.id, result.ttft, result.tokens.count)
}
```

**Streaming**: pass `onTokens` to receive each request's tokens as they
materialize — once at first token, then every decode chunk (4–8 tokens,
~150–300ms apart at phone speeds). Set `engine.decodeChunk = 1` for strict
per-token streaming at a small throughput cost (the chunked lazy graph is
worth ~7%):

```swift
let report = try engine.run(requests: requests) { id, newTokens, done in
    // called on the engine's thread; hop to your actor before touching UI
    render(id, detokenizer.append(newTokens), finished: done)
}
```

**Prefix caching**: agent fleets share a system prompt, and prefill dominates
agent workloads. Freeze the shared prefix once and every request prefills only
its suffix:

```swift
let shared = PrefixCache.longestCommonPrefix(of: prompts)   // token-level LCP
let prefix = try engine.cachePrefix(shared)                  // prefilled once
let report = try engine.run(requests: requests, prefix: prefix)
```

Measured (M-series Mac, 16-request burst, 174-token shared prefix): wall
8.4s → 7.6s and TTFT p50 −11%, with the prefix snapshot built once in 0.09s.
Correctness note: the transplanted KV state is exact, but suffix prefill uses
different chunk boundaries than full prefill, so greedy outputs can differ at
rare numeric near-ties (15/16 byte-identical in testing) — the same caveat
vLLM documents for its prefix caching.

On iOS, cap MLX's buffer cache or the recycled-buffer pool will count against
the jetsam limit:

```swift
MLX.GPU.set(cacheLimit: 64 * 1024 * 1024)
```

The `vllm-bench` executable target reproduces the benchmark scenarios (burst
and staggered-arrival) on macOS:

```sh
# Metal shaders require the Xcode build system (plain `swift build` runs
# CPU-only): build once via xcodebuild, then run the product.
xcodebuild -scheme vllm-ios -destination 'platform=macOS' \
  -configuration Release -derivedDataPath .xcbuild \
  -skipMacroValidation -skipPackagePluginValidation build
./.xcbuild/Build/Products/Release/vllm-bench --n 16 --batch 8 --scenario both
```

## What this is and isn't

**Is:** an embeddable continuous-batching engine — request queue,
arrival-time admission, batched prefill, lockstep batched decode, early exit.
Quant-agnostic (runs whatever mlx-swift-lm loads, including calibrated dynamic
quants like [OptiQ](https://huggingface.co/mlx-community/Qwen3.5-0.8B-OptiQ-4bit)).

**Isn't (yet):** an OpenAI-compatible server, a streaming token API, a
prefix cache, a sampler (greedy only), or a ragged-batch scheduler.
Roadmap, in value order:

- [x] Prefix caching (see below)
- [x] Streaming token callbacks (chunk-granularity; `decodeChunk = 1` for strict per-token)
- [ ] Sampling beyond greedy
- [ ] Per-sequence RoPE offsets + padding masks (drop the uniform-offset
      restriction; mlx-swift-lm already ships half the plumbing)

## FleetChat

`FleetChat/` is the demo app: a ChatGPT-style chat where each message fans out
to **four specialist agents** (key facts, plan, risks, alternatives) served
concurrently by the engine — batched decode, a shared-prefix cache built per
turn, streaming tokens rendering live into a 2×2 card grid, and early exit as
each agent finishes. One question, four answers streaming at once, entirely
on-device. Open `FleetChat/FleetChat.xcodeproj`, set your team, run on a
device; the model (~600 MB) downloads on first launch.

## IceBench

`IceBench/` is the iOS benchmark app behind every number above — and the
methodology is the point: it enforces thermal gates and mandatory cooling gaps
between runs, tracks thermal state per run, and measures burst duty cycles,
because an un-gated mobile LLM benchmark measures the ordering of its test
cases, not the software. It benchmarks three arms side by side: llama.cpp
(GGUF), stock single-stream MLX, and the vllm-ios engine (burst, staggered
arrivals, and a b1/2/4/8 scaling sweep), with a model library covering uniform,
mixed, and calibrated-dynamic quants. Results persist on-device and export as
JSON.

```sh
cd IceBench
./fetch-llama-xcframework.sh   # one-time: prebuilt llama.cpp Metal framework
open IceBench.xcodeproj        # set your team, run on a device
```

## Related work

| Project | What it is | iOS? | Multi-sequence batching? |
|---|---|---|---|
| [vllm-mlx](https://arxiv.org/abs/2601.19139) | Python engine on MLX | ❌ Mac | ✅ |
| mlx-lm (Python) `BatchGenerator` | Upstream Python batch generation | ❌ Mac | ✅ |
| [mlx-swift-lm PR #263](https://github.com/ml-explore/mlx-swift-lm/pull/263) | Swift continuous batching, unmerged since May 2026 | ⏳ not released | ✅ (pending) |
| [TheTom/vllm-swift](https://github.com/TheTom/vllm-swift) | Python vLLM plugin on mlx-swift | ❌ Mac | ✅ |
| [SwiftLM](https://github.com/SharpAI/SwiftLM) | MLX Swift server + iOS app | ✅ | ❌ |
| [qwen3.5-mlx-continuous-batching](https://github.com/PerhapxinLab/qwen3.5-mlx-continuous-batching) | Swift VLM server (35B) | ❌ Mac | ✅ |
| llama.cpp | GGUF runtime | ✅ | ✅ (weak Metal small-batch kernels: 1.4–2.0x at B=8 on A-series) |
| **vllm-ios** | **Embeddable Swift engine** | ✅ | ✅ (1.9–3.3x at B=8) |

## License

Apache-2.0
