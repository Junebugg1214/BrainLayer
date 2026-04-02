# BrainLayer Baseline Report

## Official Baseline

The current official baseline is `study-v1-gemini-core`.

- study bundle: `artifacts/study_runs/20260402T175429Z-study-v1-gemini-core`
- study label: `study-v1-gemini-core`
- run commit: `2cbdde9`
- protocol snapshot: `docs/study_protocol.md`
- config snapshot: `examples/model_matrix.gemini.chat.core.live.json`

This baseline is frozen and should be treated as the reference point for future `study-v2` changes.

## Model Set

The frozen baseline uses two Gemini models through the OpenAI-compatible Gemini endpoint:

- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

## Aggregate Results

### `gemini-2.5-flash`

- overall: `43/47`
- contradiction: `16/16`
- natural: `27/31`
- natural extraction: `14/16`
- natural behavior: `13/15`
- estimated total cost: `$0.0185`
- average latency: `1992.9ms`

### `gemini-2.5-flash-lite`

- overall: `39/47`
- contradiction: `14/16`
- natural: `25/31`
- natural extraction: `13/16`
- natural behavior: `12/15`
- estimated total cost: `$0.0051`
- average latency: `982.1ms`

## Pack Breakdown

### Standard

- `gemini-2.5-flash`: `19/19`
- `gemini-2.5-flash-lite`: `16/19`

### Hard

- `gemini-2.5-flash`: `14/14`
- `gemini-2.5-flash-lite`: `14/14`

### Held-Out

- `gemini-2.5-flash`: `10/14`
- `gemini-2.5-flash-lite`: `9/14`

## Main Findings

1. BrainLayer is strong on the authored and harder delayed/noisy packs.
   Both models reached perfect `hard` scores, and `gemini-2.5-flash` also reached a perfect `standard` score.

2. Held-out generalization is the real boundary.
   The main remaining weakness is not contradiction handling, which is excellent, but natural-language generalization under alternate wording.

3. `gemini-2.5-flash` is the best quality model in the frozen baseline.
   It has the best aggregate score and remains the best value at the frozen baseline frontier.

4. `gemini-2.5-flash-lite` is the speed-and-cost model.
   It is materially cheaper and faster while still staying competitive enough to matter for budget-sensitive runs.

## Reliability Notes

- `gemini-2.5-flash` showed a small number of parse failures in the frozen baseline but no hard runtime errors.
- `gemini-2.5-flash-lite` was faster and cheaper, but less stable overall and lower-scoring than `gemini-2.5-flash`.

## Frozen Boundary

This baseline intentionally stops before additional held-out-specific tuning.

In particular, later exploratory work identified held-out wording like `punchy` as a likely semantic neighbor of `brief`, but that mapping is not included in the official baseline because it would push the project toward tuning on held-out phrasing rather than preserving the held-out pack as a genuine generalization check.

Exploratory post-baseline reruns may exist in `artifacts/`, but they are not part of the official baseline unless explicitly promoted into a new study version.

## Interpretation

The baseline supports the core BrainLayer thesis:

- layered cognitive state can support strong contradiction handling
- layered cognitive state can support natural extraction and later behavior
- the main research pressure has moved from infrastructure to generalization

The baseline does not support the claim that the problem is solved. Instead, it gives a credible first reference point: strong performance on `standard` and `hard`, with `held_out` still exposing where semantic generalization weakens.

## Recommended Next Step

Treat this baseline as frozen.

From here:

1. write external/public summaries from this baseline
2. keep future system changes under a new label such as `study-v2-gemini-core`
3. compare future runs against this bundle rather than silently updating the baseline
