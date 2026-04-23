# ApproxMLIR MLSys Talk Script

Subtitle: `A 30-minute talk centered on MLIR examples, with BM25 and decode quantization as the running examples`

Presenter note:
- This script is written for an MLSys audience and assumes the talk is given in English.
- The quantization example is intentionally presented as a direct `approx.transform`-driven rewrite, because that is the clearest way to explain the ApproxMLIR design.
- For the talk, do not center the story on function substitution. Center it on the dialect, the passes, and the rewrite semantics.

## Slide 1: Title and Thesis

Script:

Hi everyone, I am going to present ApproxMLIR, an accuracy-aware compiler for compound AI systems.

The problem we are trying to solve is simple to state but surprisingly hard to engineer: modern AI systems are no longer just one model. They are pipelines that combine LLMs, retrieval, search, ranking, tools, and ordinary systems code. Every one of those components already contains approximation opportunities. We can quantize the model. We can prune retrieval. We can simplify ranking. We can switch algorithms at runtime. But today, these opportunities are fragmented across unrelated stacks.

The core thesis of ApproxMLIR is that approximation should be treated as a first-class compiler concern, with one common representation across ML and non-ML code, instead of being reimplemented independently in every frontend and runtime.

Today I will explain that idea through MLIR code, not just architecture diagrams. I will use two concrete examples throughout the talk: BM25 retrieval on the non-ML side, and decode-time kernel quantization on the LLM side.

## Slide 2: The Paper’s Running Example Is BM25 RAG

Script:

The paper motivates ApproxMLIR through a BM25-based retrieval-augmented generation pipeline.

This is a good example because it already has exactly the kind of structure that breaks current compiler flows.

The pipeline is:

- run BM25 scoring over a document corpus
- retrieve top documents
- assemble a prompt
- call the LLM to generate the final answer

And that means the system already contains both:

- non-ML code, such as BM25 and prompt assembly
- ML code, namely the generation model

So even before we talk about approximation, this is already a multi-toolchain system.

## Slide 3: Challenge 1, Fragmented Toolchains

Script:

The first challenge in the paper is fragmented toolchains.

In the BM25 RAG example, BM25 is ordinary systems code, so it may compile through Clang or Polygeist into `scf` or lower-level IR.

The LLM path is completely different. In the paper, it comes from JAX and lowers through `stablehlo` and then through IREE or XLA.

So if I want to approximate both sides of the pipeline, there is no single place where I can do that uniformly.

I end up writing one approximation mechanism for retrieval, another for the LLM, and then I still do not have a clean way to coordinate them at runtime or during tuning.

That is challenge number one: the approximations live in different worlds.

## Slide 4: Challenge 2, Prior Unification Was Too Low-Level

Script:

The second challenge is that previous attempts at unification were often too low-level.

If we wait until everything becomes LLVM IR or raw CUDA kernels, then the compiler has already lost most of the structure we actually care about.

Tensor operations become pointer arithmetic.
Matrix multiplies become loop nests.
Structured control flow becomes low-level branches and phi nodes.

So yes, LLVM gives you one common representation, but it gives it to you after most of the semantic information is already gone.

And that is exactly the wrong level if you want to express approximation strategies that depend on loops, tensors, regions, or high-level structure.

## Slide 5: Challenge 3, Static Approximation Is Too Rigid

Script:

The third challenge is that static approximation is often the wrong policy.

In compound systems, the best approximation choice depends on input characteristics and runtime state.

For BM25, maybe the retrieval score suggests we can prune aggressively for one query but not another.

For the LLM side, maybe the approximation regime should depend on context or system state.

The paper’s point is that dynamic approximation can produce better tradeoffs than a single static choice, but MLIR does not natively provide a clean mechanism for this kind of approximation management.

So challenge three is not just “how do I approximate?”

It is “how do I decide at runtime which approximation branch should be active?”

## Slide 6: Challenge 4, MLIR Still Has a Gap

Script:

The fourth challenge is subtler, and this is where the paper becomes very compiler-specific.

MLIR already gives us a multi-level IR stack, but it does not give us approximation support out of the box.

A naive approach would be to attach approximation metadata as attributes to existing ops.

For example, imagine attaching something like `approx.transform = "skip"` to an `scf.for`.

The problem is that later passes may tile, outline, split, or rewrite that loop. Once that happens, the exact op that carried the attribute may disappear, and the approximation metadata may be dropped silently.

So the issue is not just that MLIR lacks approximation syntax.

The issue is that approximation needs its own compiler-managed representation if it is going to survive the rest of the pipeline.

## Slide 7: ApproxMLIR’s Response to the Four Challenges

Script:

Now we can state the paper’s solution precisely.

ApproxMLIR responds to those four challenges with one central move:

it introduces a separate `approx` dialect that represents approximation as first-class compiler operations.

That solves the four challenges in a very direct way.

First, both the non-ML path and the ML path are brought into MLIR and then wrapped in the same approximation interface.

Second, the approximation logic stays at a level where high-level structure still exists.

Third, the dialect includes explicit approximation-management operations, so runtime-dependent selection is part of the design instead of an afterthought.

Fourth, approximation metadata is no longer a fragile attribute on someone else’s op. It lives in first-class `approx` operations that are lowered explicitly by `approx-opt`.

That is the real motivation for the dialect design.

## Slide 8: The Key Design Move: A Separate `approx` Dialect

Script:

The most important design decision in the paper is not a particular optimization. It is the decision to create a separate dialect.

Why is that important?

Because approximation metadata is not just a local attribute on one op. It has scope. It can depend on runtime state. It can contain search parameters. It can choose among multiple approximation strategies. And it has to survive transformations like tiling, bufferization, and lowering.

If we try to encode all that as attributes on existing dialect ops, we lose it when those ops get rewritten.

So ApproxMLIR introduces three key operations:

- `approx.knob` for approximation scope and autotuner interface
- `approx.decision_tree` for runtime selection logic
- `approx.transform` for the concrete approximation implementation

The conceptual split is:

- `approx.knob` and `approx.decision_tree` answer when and where to approximate
- `approx.transform` answers how to approximate

That separation is what makes the system compositional.

## Slide 5: Minimal Dialect Example

Script:

Let me show the smallest ApproxMLIR idea in code.

```mlir
%r = approx.knob(%arg0, %arg1) <{
  id = 7,
  params = "...",
  transform_types = ["bm25_subset", "decode_quantize"]
}> ({
  approx.decision_tree(%state) <{
    runtime = "get_runtime_state",
    thresholds = [0, 10, 50],
    decisions = [0, 1, 2, 3],
    transform_type = "bm25_subset"
  }>

  // exact region
  ...
  approx.yield %result
}) : (...) -> ...
```

What matters here is not the syntax itself. What matters is the compiler meaning.

This says: here is a region of exact computation. It is a legal approximation site. It has an external identity, so a tuner and a runtime can control it. And the decision of which approximation level to use can depend on runtime state.

Then, later in the pipeline, the chosen branch emits one or more `approx.transform` ops, and those are lowered by rewrite passes.

That is the core ApproxMLIR story in one slide.

## Slide 6: Example 1, Loop Perforation Walkthrough (Figure 4)

Script:

The clearest way to see how the dialect actually works end to end is the paper's Figure 4. It walks loop perforation through every stage of the toolchain. I will trace it.

Step 0: the user marks a target loop with a structured comment. This is the only thing the user writes by hand.

```c
// @approx:decision_tree {
//   state_function: get_state
//   thresholds_lower: [0]
//   thresholds_upper: [100]
//   decision_values: [0, 1]
//   transform_type: loop_perforate
// }
int kernel(int n, ...) {
  for (int i = 1; i < n; i++) {
    ...
  }
}
```

Step 1: the frontend emits an `approx.knob` op around the loop. The loop region is hoisted into the knob body. The `id` and `params` attributes are the autotuner interface.

```mlir
func.func @kernel(%arg0: i32, ...) {
  ... = approx.knob(...) <{id = 0, params = "..."}> ({
    // for loop, stride = 1
  }) : (...) -> ...
}
```

Step 2: the management op `approx.decision_tree` is emitted inside the knob body. It carries the runtime function name, the thresholds, and the per-branch decisions.

```mlir
... = approx.knob(...) <{id = 0, params = "..."}> ({
  approx.decision_tree(%arg0) <{
    transform_type = "loop_perforate",
    runtime = "get_state",
    thresholds = [...],
    decisions = [...]
  }> : (...) -> i32
  // for loop, stride = 1
}) : (...) -> ...
```

Step 3: `approx-opt` lowers `decision_tree` into ordinary MLIR control flow. A runtime call returns a state value. Threshold comparisons turn into a count, which feeds an `scf.index_switch`. Each branch contains an `approx.transform` op with a different `knob_val`.

```mlir
%0 = call @get_state(%arg0) : ...
%1 = arith.cmpi sge, %0, <thresholds> : i32
%2 = arith.select %1, <one>, <zero> : i32
... // count threshold matches into %cnt
scf.index_switch %cnt
case ... {
  approx.transform <{transform_type = "loop_perforate", knob_val = ...}>
  // for loop, stride = 1
}
```

Step 4: the runtime function `@get_state` is bound. Its body is supplied by `approx-runtime` and linked in at this stage.

Step 5: each `approx.transform` op is rewritten according to its `knob_val`. For loop perforation, that means changing the loop stride.

```mlir
case ... {
  // approximate at runtime
  %5 = scf.for %arg2 = %c1 to %4 step %c2 -> ... { ... }
  // loop, stride = 2
}
```

The takeaway is that the dialect is not just syntax. Each op has a dedicated lowering pass, and the lowering is staged: management first, then implementation. That staging is what lets one transform implementation be reused across many knobs and many call sites.

## Slide 7: Why This Design Is Better Than Attributes

Script:

This is where the separate dialect really matters.

Suppose I had attached an attribute directly to the loop, something like `approx = "subset"`.

Then some later pass tiles or outlines or canonicalizes that loop. Suddenly the exact op I annotated is gone. My approximation metadata may be dropped, duplicated incorrectly, or become ambiguous.

By contrast, when approximation is represented as first-class ops, the approximation boundary is explicit and compiler-managed. Standard MLIR passes can proceed normally, and ApproxMLIR lowers its own dialect on purpose, in dedicated passes.

That is why the dialect is not just convenient. It is necessary.

## Slide 8: Example 2, Decode Quantization

Script:

Now let me move to the ML side, and this is the example I want the audience to remember.

For LLM decode, the exact kernel often looks like a matrix multiply at the kernel level. Conceptually, in a Triton-style or TTIR-style form, the heavy op is:

```mlir
%acc = tt.dot %a, %b, %acc_init
```

For this example, the right compiler question is not just “how do I quantize the kernel?”

The right question is:

- what should be rewritten inside the kernel
- and what should be prepared outside the kernel because it should not be recomputed on every launch

That split is the clean way to explain quantization in ApproxMLIR.

## Slide 9: Kernel-Internal Rewrite

Script:

The kernel-internal part is what belongs in `approx.transform`. Everything that changes the per-launch compute path goes here, including activation quantization, because activations are input-dependent and cannot be pre-computed.

We can express the exact kernel region as:

```mlir
%out = approx.knob(%a, %b) <{
  id = 21,
  params = "...",
  transform_types = ["decode_quantize"]
}> ({
  approx.transform <{
    transform_type = "decode_quantize",
    transform_value = 1
  }>

  %acc = tt.dot %a, %b, %acc_init
  approx.yield %acc
}) : (...) -> tensor<...xf32>
```

Then the conceptual rewrite produced by `approx-opt` is:

```mlir
// activation quant is part of the rewritten compute path
%aq, %a_scale = quantize_activation(%a_bf16)

%acc_i32 = tt.dot %aq, %bq, %zero_i32
         : tensor<1xKxi8> * tensor<KxNxi8> -> tensor<1xNxi32>

%out = dequant(%acc_i32, %a_scale, %b_scale)
     : tensor<1xNxi32> -> tensor<1xNxbf16>
```

Three things to take away:

- the heavy op changed from bf16 matmul to int8 matmul
- activation quantization is hoisted into the rewrite, not left as a separate op outside
- the weight side only appears as pre-computed operands `%bq` and `%b_scale`, which is what the next slide is about

One useful consequence of this split is kernel sharing. If two projections see the same input, for example `gate_proj` and `up_proj` both reading the post-RMSNorm activation, the rewrite can compute `%aq` once and feed both matmuls from it. That is only expressible if activation quant lives inside the transform region.

## Slide 10: Kernel-External Preparation

Script:

Now the part that stays outside the kernel. The criterion is sharp: if an artifact is reusable across launches, or reusable across multiple kernels, it belongs outside.

Weight quantization is the clearest case. Weights are computed once at model load and read by every decode step for the lifetime of the process. Re-quantizing them on every launch would be wasteful by construction.

```mlir
// once per model load, outside any kernel
%bq, %b_scale = pre_quantize_weights(%b_bf16)
```

These operands are then referenced by the rewritten kernel as ordinary inputs. The rewrite itself does not contain a `quantize_weights` op; it only consumes the pre-computed result.

So the design rule is:

- reusable model-side information, such as quantized weights and their scales, stays outside the kernel as precomputed artifacts
- anything that changes the per-launch compute path, including activation quantization, is rewritten into the kernel by `approx.transform`

This split is not cosmetic. It is what lets multiple projections share one activation-quant result, and it is what makes weight quantization a one-time cost rather than a per-step cost.

## Slide 11: Where AWQ Fits

Script:

This also gives us a clean way to explain future extensions such as AWQ.

AWQ needs offline information, for example activation statistics collected from representative data.

That information is not kernel-side computation. It is kernel-external metadata.

So the natural ApproxMLIR story is:

- collect calibration or activation statistics outside the kernel
- encode them as approximation metadata or artifact-side parameters
- pass them into the rewritten kernel through the quantized weight representation or associated scales

In other words, AWQ does not change the architectural split. It just enriches the data that flows from kernel-external preparation into the kernel-internal rewrite.

## Slide 12: Why Runtime Still Matters

Script:

At this point, it is tempting to think ApproxMLIR is just a static rewrite engine. It is not.

The paper is explicit that runtime is central. This is where `approx-runtime` enters.

There are two runtime roles:

First, runtime chooses the approximation level. For example, in BM25 it may decide how aggressively to subset documents. In decode it may decide whether to stay exact, use one quantization regime, or use another.

Second, runtime supplies the state used by approximation management, through the functions bound to `approx.decision_tree`.

So the clean mental model is:

- `approx-opt` rewrites the computation
- `approx-runtime` decides when and which branch is active

That separation is what makes the system useful in real workloads.

## Slide 13: Two Examples, One Uniform Interface

Script:

What I like about ApproxMLIR is that these two examples are genuinely different:

- BM25 is ordinary control-flow-heavy systems code
- decode quantization is a tensor-kernel rewrite problem

But the interface is still the same.

In both cases:

- a region is wrapped by `approx.knob`
- runtime policy is expressed by `approx.decision_tree`
- concrete implementation is introduced by `approx.transform`
- and `approx-opt` lowers those transforms into rewrites that are legal for the target dialect

That is what the paper means by a unified approximation interface.

## Slide 14: What the Toolchain Does End to End

Script:

Let me summarize the toolchain flow in compiler terms.

Step one: the frontend emits normal MLIR plus `approx` ops.

Step two: the autotuner sees knobs, not backend-specific implementation details. That is important. The search framework does not need to understand `stablehlo`, `scf`, `linalg`, or Triton internals. It only needs to reason about the approximation space exposed by the `approx` dialect.

Step three: `approx-opt` lowers management operations such as `decision_tree`, binds runtime hooks, and applies transform rewrites.

Step four: normal backend compilation continues. At the end, `approx-runtime` orchestrates the final dynamic behavior and gives us a Pareto frontier rather than a single static build.

So the output of ApproxMLIR is not just one approximated program. It is a family of controlled approximations that the user can choose from.

## Slide 15: Why This Matters for MLSys

Script:

I think the MLSys contribution here is deeper than “we added another quantization pass”.

The important system contribution is that ApproxMLIR gives us a place in the compiler stack where end-to-end approximation is representable, optimizable, and searchable across different components.

That changes the unit of optimization:

- not one kernel
- not one model
- but the whole compound AI system

And once we have that, we can express heterogeneous approximation strategies in one framework:

- retrieval reduction in BM25
- kernel-side decode quantization
- dynamic runtime selection
- and, more generally, coordinated approximation across ML and non-ML code

## Slide 16: Evaluation Setup

Script:

Let me now be concrete about the evaluation, because this is where the paper validates the design.

ApproxMLIR is evaluated on two scales of workloads.

First, five individual kernels.

Second, three compound AI systems:

- LLM + RAG with BM25
- LLM + RAG with knowledge-base retrieval
- LLM + tool invocations

The LLM is Gemma 3, using 1B and 4B variants. The RAG benchmarks use the NQ dataset, and the corpus has 90,011 documents. The evaluation set is five times larger than the tuning set, and the two are disjoint, so the paper is explicitly checking whether tuning generalizes.

Another useful detail is the size of the approximation spaces. These are not toy spaces. For example, the LLM + tools benchmark has 12 knobs and about 1.7 times 10 to the 25 possible configurations. So the compiler is searching a genuinely large end-to-end space.

## Slide 17: QoS Metrics in the Paper

Script:

The paper also does the right thing on quality metrics.

For compound AI systems, accuracy is defined at the task level as question score: a question gets a 1 if the model’s generated answer contains the correct short answer, and 0 otherwise.

For non-ML kernels such as kmeans and lavaMD, the paper uses normalized L2 accuracy.

For ranking-centric workloads such as pagerank and the RAG retrieval systems, it uses Rank-Biased Overlap, or RBO, with persistence parameter 0.95.

So ApproxMLIR is not forcing one fake notion of accuracy across all components. It uses workload-appropriate QoS definitions, which is exactly what a systems paper on compound AI should do.

## Slide 18: Figure 7, End-to-End Evaluation Result

Script:

The first result I would actually point at on the slide is Figure 7.

This figure has three bar charts, one for each QoS-loss budget: 3 percent, 6 percent, and 9 percent. In each chart:

- the dashed black line is the exact baseline at 1.00x
- the red bar is static approximation
- the green bar is ApproxMLIR’s fine-grained dynamic approximation

And there are three end-to-end compound systems:

- LLM + RAG with BM25
- LLM + RAG with knowledge-base retrieval
- LLM + tools

The visual message is not “everything improves equally.” The visual message is more interesting.

The strongest win is clearly LLM + RAG with knowledge-base retrieval. That is where the green bar separates most strongly from the red bar and from the 1.00x baseline.

The exact numbers the paper highlights are:

- 2.64x at 3 percent QoS loss
- 2.64x at 6 percent QoS loss
- 3.04x at 9 percent QoS loss

By contrast, the LLM + tools benchmark is much harder. In the figure, those bars sit much closer to 1.00x, which tells us this benchmark has less exploitable approximation headroom under the current search budget.

And LLM + BM25 sits in between: it improves, but it is not as dramatic as the KB case.

That is exactly the kind of evaluation result I want in a systems paper. It does not pretend all workloads respond the same way.

## Slide 19: Figure 8, Compound-System Pareto Frontiers

Script:

Figure 8 is where the paper shows the real systems argument, not just isolated speedup numbers.

This figure plots the Pareto frontiers for the three compound AI systems:

- `LLMbm25`
- `LLMkb`
- `LLMtool`

On these plots:

- x-axis is QoS loss, lower is better
- y-axis is execution time, lower is better
- each point is one configuration explored by the autotuner
- green is fine-grained dynamic approximation
- red is static approximation

The important visual pattern is that the dynamic frontier is not just slightly lower. It is usually both:

- lower
- and denser

Lower means better tradeoff points. Denser means more operating points for the user.

That second point matters. ApproxMLIR is not only claiming “we found one better configuration.” It is claiming “the dialect design and runtime support produce a better tradeoff surface.”

And in Figure 8, especially for the compound systems, that claim is visually convincing.

## Slide 20: Figure 9, Kernel-Level Pareto Frontiers

Script:

Figure 9 repeats the same analysis, but now at the individual-kernel level:

- `lavaMD`
- `kb`
- `bm25`
- `pagerank`
- `kmeans`

This figure is important because it shows that the benefit of fine-grained dynamic approximation is not limited to one level of granularity.

Some kernels show very strong separation between dynamic and static frontiers. Some show smaller gains. But the overall pattern remains: dynamic approximation gives either better frontier points, more frontier points, or both.

So when I connect this back to the talk examples, the message is:

- BM25 is not only a motivating example, it is also a measured kernel-level workload
- and the same ApproxMLIR abstraction scales from kernels to full compound systems

That is one of the strongest aspects of the paper.

## Slide 21: Compiler Cost and Why the Paper Says It Is Acceptable

Script:

The paper also reports compiler overhead.

ApproxMLIR takes on average:

- 120 seconds to compile ML kernels
- 5 seconds to compile non-ML kernels

The autotuning budget is 12 hours for compound AI systems and 2 hours for the kernel benchmarks.

The paper’s argument is that this overhead is acceptable because the output is not a single binary. The output is a tuned tradeoff curve, and that curve generalizes from tuning to evaluation.

The paper also reports that the difference between tuning and evaluation speedups is up to:

- 18.6 percent for LLM + BM25
- 19.5 percent for LLM + KB
- 3.7 percent for LLM + tools

So the systems claim is not that compilation is free. The claim is that the compile-and-tune cost is justified because the tuned frontier transfers reasonably well to held-out evaluation inputs.

## Slide 22: How the Evaluation Connects Back to the Two Running Examples

Script:

Let me reconnect the evaluation to the two examples from this talk.

For BM25, the evaluation shows two things:

- it matters at the end-to-end compound-system level in Figure 7 and Figure 8
- and it also appears directly as a kernel-level frontier in Figure 9

So BM25 is not just a story device in the introduction. It is one of the places where the ApproxMLIR abstraction is actually exercised in the results.

For decode quantization, the lesson is architectural rather than empirical in this paper. The same abstraction that lets us express BM25 pruning also gives us a principled place to express kernel-side decode quantization:

- approximation scope with `approx.knob`
- runtime control with `approx.decision_tree`
- concrete kernel rewrite with `approx.transform`

So in the talk, BM25 anchors the paper’s measured evaluation, and decode quantization shows how the exact same compiler abstraction extends naturally to a practical LLM-kernel approximation example.

## Slide 23: Closing

Script:

Let me close with one sentence.

ApproxMLIR is not mainly about one approximation technique. It is about giving approximation a proper compiler abstraction.

Once we do that, BM25 pruning and decode quantization stop looking like unrelated hacks and start looking like what they really are: structured rewrites over a common IR, coordinated by one compiler and one runtime.

Thank you.

## Optional Backup Slide A: A More Concrete Quantization Rewrite

Use this if someone asks for a more operational view.

Script:

If you want a more concrete view of the decode rewrite, the intended compiler picture is:

```mlir
// exact
%0 = tt.dot %a_bf16, %b_bf16, %acc0

// after approx transform
%aq, %sa = approx.runtime.quantize_activation %a_bf16
%bq, %sb = approx.runtime.quantize_weight %b_bf16
%1 = tt.dot %aq, %bq, %acc_i32
%2 = approx.runtime.dequantize_matmul %1, %sa, %sb
```

The compiler owns the structural rewrite. The runtime owns the dynamic scale-producing logic. That is the division of labor.

## Optional Backup Slide B: A More Concrete BM25 Rewrite

Use this if someone asks how ApproxMLIR handles non-ML kernels.

Script:

For BM25, a good way to think about the rewrite is:

```mlir
// exact
scf.for %i = %c0 to %num_docs step %c1 {
  %s = call @bm25_score_doc(...)
}

// approximated
%limit = call @runtime_doc_budget(%state)
scf.for %i = %c0 to %limit step %c1 {
  %s = call @bm25_score_doc(...)
}
```

The point is not that this specific rewrite is magical. The point is that the approximation is represented in the same framework as the quantized decode rewrite, even though the kernels live in very different worlds.

## Optional Backup Slide C: Activation Quantization Inside the Rewritten Kernel

Use this if someone asks what the activation-side quantization logic actually looks like.

Script:

The activation path is input-dependent, so it belongs on the kernel side of the split.

Conceptually, the rewritten kernel computes one scale per activation row, then uses that scale to quantize the input before the integer dot:

```mlir
%v = ... load bf16 activation tile ...
%v_f32 = arith.extf %v : tensor<...xbf16> to tensor<...xf32>
%abs = math.absf %v_f32
%amax = "tt.reduce"(%abs) ...
%scale = arith.divf %amax, %c127
%q = arith.mulf %v_f32, %inv_scale
%q_rounded = ... round ...
%q_i8 = arith.fptosi %q_rounded : tensor<...xf32> to tensor<...xi8>
```

So the activation path has a very clear structure:

- convert to a higher-precision working type
- compute a local scale
- round and clamp
- convert to `int8`

That is the part of quantization that is naturally tied to the per-launch kernel computation.

## Optional Backup Slide D: INT8 Dot and Dequant Epilogue

Use this if someone asks what the rewritten compute path looks like after activation and weight quantization are available.

Script:

Once the kernel has quantized activations and pre-quantized weights, the central compute path becomes an integer dot followed by a dequantization epilogue.

At the IR level, the important pattern is:

```mlir
%acc_i32 = tt.dot %aq, %bq, %acc0
  : tensor<...xi8> * tensor<...xi8> -> tensor<...xi32>

%acc_f32 = arith.sitofp %acc_i32 : tensor<...xi32> to tensor<...xf32>
%out_f32 = arith.mulf %acc_f32, %a_scale
%out_f32_2 = arith.mulf %out_f32, %b_scale
%out = arith.truncf %out_f32_2 : tensor<...xf32> to tensor<...xbf16>
```

So the rewritten kernel is structurally:

- integer accumulation for the heavy dot
- floating-point rescaling at the end
- final truncation back to the output type

That is the core kernel-side quantized matmul story.

## Optional Backup Slide E: Weight Quantization and Packing Outside the Kernel

Use this if someone asks what stays outside the kernel and why.

Script:

The weight path has a different structure from the activation path.

Weights are reusable across decode steps, so the useful work happens before the kernel launches:

```mlir
// conceptual artifact-side preparation
%b_f32 = arith.extf %b_bf16 : tensor<KxNxbf16> to tensor<KxNxf32>
%b_abs = math.absf %b_f32
%b_scale = ... reduce over output columns or groups ...
%bq = ... round, clamp, convert to int8 ...
```

Optionally, the quantized weights can also be packed into a layout chosen for the target kernel.

So the weight path is best understood as:

- one-time or infrequent preprocessing
- producing quantized weight data plus scales
- then handing those precomputed tensors to the rewritten kernel

This is why the talk separates quantization into:

- activation quantization inside the rewritten compute path
- weight quantization and packing outside the kernel as reusable artifacts

## Presenter Checklist

- Spend most of the time on Slides 4 through 10.
- Keep returning to the same sentence: `approx.knob` chooses scope, `approx.decision_tree` chooses policy, `approx.transform` chooses implementation.
- When presenting quantization, use the transform-rewrite narrative, not the substitution implementation narrative.
- When presenting runtime, emphasize that activation quantization is a runtime service attached to a compiler rewrite, not a separate ad hoc system.
- If time runs short, compress Slides 13 through 19 into two slides:
  one slide for setup plus headline numbers, and one slide for Pareto plus compiler cost.
