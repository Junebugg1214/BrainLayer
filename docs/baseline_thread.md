# BrainLayer Baseline Thread

## Post 1

We’ve been testing a simple idea: AI agents may need more than “memory.” They may need a lightweight cognitive state, a `BrainLayer`, with working state, beliefs, autobiographical continuity, procedures, consolidation, and forgetting.

Our first frozen baseline is now in.

## Post 2

Frozen baseline: `study-v1-gemini-core`

Models:

- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

Official bundle:

- `artifacts/study_runs/20260402T175429Z-study-v1-gemini-core`

## Post 3

Headline results:

- `gemini-2.5-flash`: `43/47`
- `gemini-2.5-flash-lite`: `39/47`

On the harder delayed/noisy pack, both models hit `14/14`.

That’s a strong signal that the BrainLayer architecture is doing real work on continuity, revision, and reuse.

## Post 4

The most interesting part is where it *didn’t* fully solve the problem.

Held-out generalization is still the frontier:

- `gemini-2.5-flash`: `10/14`
- `gemini-2.5-flash-lite`: `9/14`

So the limiting factor has moved away from basic storage/retrieval mechanics and toward semantic generalization under new wording.

## Post 5

That’s exactly why we froze the baseline here.

We did **not** keep patching away every held-out wording miss.

If you tune each held-out phrase into the runtime, you stop measuring generalization and start optimizing to the test.

An imperfect benchmark result is more useful than a fake perfect one.

## Post 6

The current read:

- contradiction handling looks strong
- natural-memory extraction from ordinary dialogue is workable
- layered state looks more promising than “memory as a note store”
- held-out wording variation remains the main open problem

That’s a much more interesting result than “we got 100%.”

## Post 7

The strongest baseline model is `gemini-2.5-flash`.

The cheapest and fastest is `gemini-2.5-flash-lite`.

So the tradeoff is already visible:

- higher ceiling vs lower cost/latency
- both supported by the same BrainLayer runtime

## Post 8

The takeaway:

Agents may not just need better memory. They may need a `brain layer`.

We now have a frozen baseline for that claim, and every future change can be measured against it instead of moving the goalposts.

Figure: `docs/figures/study-v1-gemini-core-overview.svg`

## Single-Post Version

BrainLayer baseline is in. We tested whether agents need more than “memory” and instead benefit from a lightweight cognitive state: working state, beliefs, autobiographical continuity, procedures, consolidation, and forgetting. Frozen baseline `study-v1-gemini-core`: `gemini-2.5-flash` scored `43/47`, `gemini-2.5-flash-lite` scored `39/47`, both hit `14/14` on the hard pack, and held-out wording remains the real frontier. Memory may not be enough. Agents may need a brain layer.
