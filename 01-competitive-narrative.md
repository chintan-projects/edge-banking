# Competitive Narrative: Defending Liquid vs. Open-Source Small Models

## Executive Answer

RBC is not choosing a chatbot model. RBC is choosing the control plane for a banking assistant: the local intelligence layer that decides whether a customer query can be handled on-device, should move to an on-prem model, should be clarified, should be blocked, or should go to a human.

Open-source small models are a valid baseline. The reason to evaluate Liquid is that LFM is built for the exact physics of this control plane: low-latency local inference, low memory pressure, rapid customer-specific retraining, and many small decisions composed into auditable policy. A generic model can classify a question. The Liquid argument is that a purpose-built edge model plus a multi-head Decision Layer scales better when the router becomes a regulated banking system.

## The First-Principles Argument

An on-device banking router is called on every customer turn. That creates four hard constraints:

1. **Latency:** the router must be fast enough to run before every response without making the app feel cloud-bound.
2. **Memory:** it must fit inside the mobile app's real memory budget, not just a benchmark harness.
3. **Control:** it must emit stable, inspectable signals that banking policy can compose and audit.
4. **Adaptability:** it must keep changing as RBC learns the real use case, policy taxonomy, escalation rules, and risk boundaries.

This is where the architecture matters. A single three-way classifier (`local`, `on_prem`, `human`) is fine for a first benchmark. But as soon as RBC adds privacy, fraud risk, vulnerable-customer handling, ambiguity, product eligibility, confidence thresholds, and audit logging, a single label becomes brittle. The scalable shape is a **Decision Vector**: separate typed signals for intent, complexity, risk, privacy, confidence, and escalation reason, composed by deterministic software policy.

## Why LFM Is Fit For Edge Devices

LFM2 was not designed as a smaller copy of a cloud transformer. The LFM2 technical report says the architecture was selected with hardware-in-the-loop search under edge latency and memory constraints, using a hybrid backbone with gated short convolutions plus a smaller number of grouped-query attention blocks. That matters because local devices are memory-bandwidth constrained, and attention-heavy models pay more cost as prompt length grows.

For LFM2.5-350M specifically, the public model card lists:

- 350M parameters.
- 16 layers: 10 double-gated LIV convolution blocks and 6 GQA blocks.
- 32,768 token context length.
- 28T tokens of training.
- Under 1 GB memory in published edge configurations.
- Recommended use cases including data extraction, structured outputs, and tool use.

Liquid's public LFM2.5-350M benchmark reports local inference across desktop, mobile, and constrained edge devices, including iPhone 13 Mini, Pixel 6a, Raspberry Pi 5, Snapdragon, AMD, and Apple Silicon configurations. RBC should not take those numbers as a substitute for its own benchmark, but they are credible evidence that LFM2.5-350M is built for the deployment surface RBC is evaluating.

## Third-Party Edge Signal From ZETIC

ZETIC data helps because it is third-party, device-level evidence against the exact model families customers keep naming: LFM2.5-350M, Qwen3-0.6B, and Gemma 270M. It should be used as **edge performance headroom**, not as RBC task accuracy.

ZETIC's documentation defines:

- `RUN_AUTO`: default mode that lets the runtime choose a target model from metadata.
- `RUN_SPEED`: prioritizes lower latency.
- `RUN_ACCURACY`: prioritizes better score or lower loss and may be slower.

From the ZETIC screenshots provided for the RBC/MSFT discussion:

| Device | LFM2.5-350M Auto TPS | Qwen3-0.6B Auto TPS | Gemma 270M Auto TPS | Signal |
| --- | ---: | ---: | ---: | --- |
| iPhone 16 Pro | 177.96 | 65.06 | 123.89 | LFM is about 2.7x Qwen and about 1.4x Gemma on Auto throughput. |
| iPhone 16 | 176.97 | 48.65 | not shown | LFM is about 3.6x Qwen on Auto throughput. |
| Galaxy S25 Ultra | 144.99 | not shown | 101.21 | LFM is about 1.4x Gemma on Auto throughput. |

The more important customer-facing version is the conservative mode comparison:

| Device | LFM2.5-350M Accuracy-mode TPS | Qwen3-0.6B Accuracy-mode TPS | Gemma 270M Accuracy-mode TPS | Signal |
| --- | ---: | ---: | ---: | --- |
| iPhone 16 Pro | 69.72 | 18.58 | 39.20 | LFM is about 3.8x Qwen and about 1.8x Gemma. |
| iPhone 16 | 73.04 | 18.78 | not shown | LFM is about 3.9x Qwen. |
| Galaxy S25 Ultra | 76.05 | not shown | 47.50 | LFM is about 1.6x Gemma. |

The correct claim:

> Third-party ZETIC device data suggests LFM2.5-350M has materially more local throughput headroom than comparable Qwen and Gemma candidates on current iPhone and Samsung devices. That does not prove RBC classifier quality, but it makes the multi-head Decision Layer thesis more credible: if the router grows from one decision to many, LFM appears to leave more edge budget for the full control layer.

## Why The Classifier Backbone Gives Liquid An Advantage

For a router, generation is the wrong default primitive. The router does not need to write prose. It needs to produce stable decisions.

A classifier head is a small projection from a backbone representation into a known label space. Once the local backbone pass has produced the representation, each additional head is cheap. That is the compounding move:

- A generative router pays for token generation and parsing.
- Separate fine-tuned adapters pay repeated model or adapter costs.
- A shared multi-head classifier pays the backbone once, then emits several typed signals.

In the Liquid RBC POC, that pattern was not theoretical. The POC used one shared classification adapter for 14 sequence heads. The recorded POC outcomes were 95.7% mean primary metric across those heads, about 52 ms multi-head latency for routes that previously required multiple classifier passes, and classification adapter footprint reduced from about 320 MB to about 23 MB.

That does not prove RBC production readiness. It does prove the architectural scaling curve Liquid wants RBC to test: one head is comparable; seven heads is where shared LFM representations start to matter.

## Why LFM Lets The Architecture Scale

The Decision Layer scales along three axes:

1. **More decisions per turn:** adding a new policy axis should be a new head or policy rule, not a rewrite of one giant route taxonomy.
2. **More customer-specific adaptation:** RBC can shift the learned distribution toward its actual query mix, escalation rules, and product vocabulary.
3. **More deployment surfaces:** the same architecture should work across iOS, Android, desktop, on-prem, and embedded environments rather than being locked to one platform runtime.

Liquid's defensible claim is not "open source cannot do this." It is: open-source weights give RBC a starting model, while Liquid brings the model architecture, training loop, edge deployment stack, and engineering team that can adapt the model in lockstep with RBC's ML team.

The right executive framing is:

> If this is one static classifier, use the cheapest model that wins RBC's benchmark. If this is the control plane for a regulated banking assistant, optimize for the architecture that keeps getting better as RBC adds policy, risk, privacy, and product intelligence.

## Fine-Tuning, Forgetting, And "Lobotomization"

We should not claim that LFM fine-tuning magically avoids catastrophic forgetting. No responsible ML team will accept that.

What we can credibly say:

- Generic open-source fine-tuning is often post-hoc optimization on a general-purpose architecture. Timelines and regressions can be unpredictable unless the team owns strong eval discipline.
- Liquid's operating model is to shape the model, training data, and post-training process around the target capability and hardware.
- The goal is controlled distribution shift: improve the customer task while guarding against regressions on relevant retained capabilities.
- In the RBC POC, a banking-intent data regression was caught by a held-out probe set and corrected with a stronger v2 dataset. That is the behavior RBC should want from a vendor: benchmark-gated iteration, not blind fine-tuning.
- If approved for external use, Liquid has a recent enterprise benchmark example where a targeted retraining effort improved ActivityNetQA from about 39% to 55% in two weeks with no regression on POPE, RealWorldQA, and COCO-Cap. Use this only if the team is comfortable sharing it.

The phrase to use with RBC:

> We do not promise that fine-tuning is risk-free. We promise to run it like an engineering program: target capability, customer data, regression gates, side-by-side benchmarks, and direct collaboration with RBC's ML team.

## What To Say When RBC Asks "Why Not Gemma, Qwen, Or AFM?"

Say this:

> You should benchmark them. If the task is one stable classifier, an open-source model may be enough. But your own description already points to a control layer: classify by type, complexity, and risk, then route to local, on-prem, or human. That is a multi-signal decision system. Liquid's advantage is that LFM was designed for efficient edge inference, and our POC shows the multi-head version can turn many local decisions into one shared model pass plus auditable policy. The benchmark should test that full system, not just one label.

Then propose the evaluation:

- One-head benchmark: basic route classification.
- Four-head benchmark: route, complexity, risk, confidence.
- Eight-head benchmark: add privacy, escalation reason, product scope, and ambiguity.
- Measure accuracy, latency, peak memory, bundle size, route correctness, calibration, and behavior under policy changes.

## What It Would Take To Train And Bench Qwen/Gemma Fairly

A fair challenger is feasible, but it is not just "load Qwen in the app." The classifier head trained for LFM cannot be reused on Qwen or Gemma because the hidden-state geometry is different. Even if Qwen3-0.6B and LFM2.5-350M both expose 1024-dimensional hidden states, the classifier weights are tied to the backbone representation they were trained on.

Minimum fair Qwen/Gemma benchmark:

1. Use the same RBC taxonomy, train split, eval split, and held-out probe set.
2. Train a Qwen/Gemma-specific shared LoRA plus the same sequence heads.
3. Export the base model, shared adapter, and head binaries into the same app measurement harness.
4. Run the same one-head, four-head, eight-head, and full multi-head scenarios.
5. Measure per-head accuracy, route-policy correctness, p50/p95 latency, peak RSS/physical footprint, model bundle size, adapter size, cold-start/load time, and behavior under policy changes.

Estimated lift:

| Benchmark | What it proves | Approximate lift |
| --- | --- | --- |
| Base footprint only | File size, load memory, peak RSS, token throughput, embedding-pass latency | 1-3 days per model if GGUF/runtime support is ready |
| Offline classifier quality | Whether Qwen/Gemma can learn the same labels from the same data | 3-5 days for one head; 1-2 weeks for several heads |
| In-app single-head router | Basic local classifier comparison on the actual device path | About 1 week |
| In-app four/eight-head Decision Vector | Realistic comparison against RBC's expected router shape | 1-2+ weeks |
| Full 14-head shared-adapter replica | True apples-to-apples version of the POC architecture | 2-4+ weeks |

For Qwen3-0.6B specifically, the benchmark is mechanically plausible because its hidden size is 1024, matching the existing LFM head format. The training script still needs to be generalized from `Lfm2Model` to a Qwen model class or `AutoModel`, and the LoRA target modules need to be changed to Qwen's projection names. Gemma is similar in concept but needs its own hidden size, tokenizer, LoRA targets, export path, and runtime validation.

## Do We Need Fine-Tuning To Measure Memory?

No, not for the base-model footprint.

RBC can measure the following without fine-tuning:

- Base model file size.
- Quantized model size.
- App bundle impact.
- Model load time.
- Cold-start memory.
- Peak RSS / iOS physical footprint during load and dummy inference.
- Token throughput on identical prompts.
- Embedding-pass latency for classifier-style inference.

Fine-tuning is required only if the claim being tested is **classifier quality or shared-head scaling**:

- Can Qwen/Gemma hit the same route accuracy?
- Can they support four or eight heads without unacceptable latency?
- Does a shared adapter preserve quality across all heads?
- Does the adapter footprint stay competitive?
- Does the route policy produce the same decisions?

The clean way to stage this is:

1. **Stage A - footprint bakeoff without fine-tuning:** LFM vs Qwen vs Gemma, same devices, same quantization tier, same prompts, measure file size, load time, peak memory, and TPS.
2. **Stage B - one-head classifier bakeoff:** train one comparable route head and measure accuracy plus latency.
3. **Stage C - multi-head Decision Layer bakeoff:** train four/eight heads and measure whether the architecture still fits the device envelope.

This is the right answer to customers asking for apples-to-apples: we first test the physics, then the classifier, then the full decision system.

## Proof Points We Can Cite

- [Public LFM2 technical report](https://arxiv.org/abs/2511.23404): hardware-in-the-loop architecture search under edge latency and memory constraints; hybrid gated short convolutions plus GQA; up to 2x faster CPU prefill/decode than similarly sized models.
- [Public LFM2.5-350M model card](https://huggingface.co/LiquidAI/LFM2.5-350M): 350M parameters, 10 LIV convolution blocks plus 6 GQA blocks, 32K context, 28T training tokens, under 1 GB memory in published configurations, structured output/tool-use fit.
- [Public LFM2.5-350M benchmark](https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind): local inference numbers across AMD, Snapdragon, Apple Silicon, iPhone 13 Mini, Pixel 6a, and Raspberry Pi 5.
- [Public Nanos announcement](https://www.liquid.ai/press/liquid-unveils-nanos-extremely-small-foundation-models-that-match-frontier-model-quality--running-directly-on-everyday-devices): task-specialized 350M extraction model outperforming Gemma 3 4B on structured extraction.
- [Public Shopify announcement](https://www.liquid.ai/blog/liquid-ai-announces-multi-year-partnership-with-shopify-to-bring-sub-20ms-foundation-models-to-core-commerce-experiences): production search deployment under 20 ms and public claim of 2-10x faster inference than popular open-source models on specific production-like workloads.
- [Public Mercedes-Benz announcement](https://www.liquid.ai/press/liquid-ai-and-mercedes-benz-partner-to-scale-embedded-in-car-intelligence): multi-year partnership for embedded, on-device intelligence, emphasizing fast, private, local AI without dependence on the cloud.
- [ZETIC LLM inference modes](https://docs.zetic.ai/llm-inference/inference-modes): third-party documentation for `RUN_AUTO`, `RUN_SPEED`, and `RUN_ACCURACY` mode semantics.
- [ZETIC key concepts](https://docs.zetic.ai/introduction/key-concepts): third-party documentation for performance-adaptive deployment and benchmarking across 200+ physical devices.
- Internal RBC POC: 14 sequence heads, 95.7% mean primary metric, about 52 ms multi-head latency, about 320 MB to about 23 MB classification footprint.
