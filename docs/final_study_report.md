# BrainLayer Final Study Report

## Scope

This report integrates the frozen `study-v2` and `study-v3` findings into one final artifact.

It answers a narrower and more defensible question than the original project framing:

`Which parts of BrainLayer show the strongest replicated value, how broad is that value across providers, and which parts of the architecture still have not clearly earned their keep?`

This report should be read as the current best synthesis of the project, not as a claim that the architecture is finished.

## Core Question

The project started from a simple idea:

`Agents may need more than memory retrieval. They may need a lightweight cognitive state.`

In this repo, that cognitive state is implemented as `BrainLayer`, which combines:

- working state
- autobiographical continuity
- beliefs
- procedures
- consolidation
- forgetting

The study goal was to determine which of those pieces actually matter under live model evaluation.

## Reference Artifacts

### Study-V2 Baseline And Ablations

- [docs/study_v2_report.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v2_report.md)
- [docs/study_v2_ablation_report.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v2_ablation_report.md)
- [docs/study_v2_cross_provider_report.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v2_cross_provider_report.md)

### Study-V3 Stress Findings

- [docs/study_v3_stress_report.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v3_stress_report.md)
- [docs/study_v3_consolidation_note.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v3_consolidation_note.md)
- [docs/study_v3_forgetting_note.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v3_forgetting_note.md)
- [docs/study_v3_forgetting_cross_provider_memo.md](/Users/marcsaint-jour/Documents/New%20project/docs/study_v3_forgetting_cross_provider_memo.md)

## Main Result

The strongest and most replicated result in the entire project is:

`Working state and autobiographical continuity are the most important BrainLayer components.`

That pattern held across:

- Gemini
- Anthropic
- the stronger `study-v2` baseline setup
- the cross-provider ablation sweep

This is the clearest positive result in the repo.

## Study-V2: What Held Up

### Frozen Gemini Baseline

On the frozen post-patch `study-v2` reference set:

`gemini-2.5-flash`

- `brainlayer_full`: `90/95`, `92/95`, `91/95`
- `structured_no_consolidation`: `86/95`, `84/95`, `89/95`
- three-run average gap: `+4.7`

`gemini-2.5-flash-lite`

- `brainlayer_full`: `85/95`, `87/95`, `85/95`
- `structured_no_consolidation`: `81/95`, `86/95`, `87/95`
- three-run average gap: `+1.0`

This supports a strong/stable full BrainLayer win on `gemini-2.5-flash` and a smaller, noisier result on `gemini-2.5-flash-lite`.

### Frozen Ablation Read

On the frozen `study-v2` ablations:

`gemini-2.5-flash`

- `brainlayer_full`: `91/95`
- `brainlayer_no_consolidation`: `90/95`
- `brainlayer_no_forgetting`: `92/95`
- `brainlayer_no_autobio`: `79/95`
- `brainlayer_no_working_state`: `78/95`

`gemini-2.5-flash-lite`

- `brainlayer_full`: `87/95`
- `brainlayer_no_consolidation`: `87/95`
- `brainlayer_no_forgetting`: `87/95`
- `brainlayer_no_autobio`: `75/95`
- `brainlayer_no_working_state`: `77/95`

The clear takeaway from `study-v2` is:

- removing `autobio` hurts a lot
- removing `working_state` hurts a lot
- removing `consolidation` barely changes totals
- removing `forgetting` is neutral or slightly positive on that frozen setup

### Cross-Provider Replication

That component pattern replicated on Anthropic:

`claude-sonnet-4.5`

- `brainlayer_full`: `90/95`
- `brainlayer_no_consolidation`: `90/95`
- `structured_no_consolidation`: `90/95`
- `brainlayer_no_forgetting`: `89/95`
- `brainlayer_no_working_state`: `79/95`
- `brainlayer_no_autobio`: `78/95`

`claude-haiku-4.5`

- `brainlayer_full`: `86/95`
- `brainlayer_no_consolidation`: `88/95`
- `structured_no_consolidation`: `87/95`
- `brainlayer_no_forgetting`: `87/95`
- `brainlayer_no_working_state`: `77/95`
- `brainlayer_no_autobio`: `78/95`

So the strongest `study-v2` claim is not that the whole BrainLayer stack wins everywhere. It is that `autobio` and `working_state` are the most stable, most replicated contributors.

## Study-V3: What Changed

`study-v3` was created because `study-v2` showed that `consolidation` and `forgetting` had not yet earned their place.

The goal of `study-v3` was to stress those components directly rather than assume they matter.

### Forgetting

The updated forgetting read is:

`gemini-2.5-flash`

- `brainlayer_full`: `8/8`
- `brainlayer_no_forgetting`: `6/8`
- `structured_no_consolidation`: `5/8`

`claude-haiku-4.5`

- `brainlayer_full`: `6/8`
- `brainlayer_no_forgetting`: `4/8`
- `structured_no_consolidation`: `5/8`

`claude-sonnet-4.5`

- `brainlayer_full`: `8/8`
- `brainlayer_no_forgetting`: `8/8`
- `structured_no_consolidation`: `7/8`

`gpt-4.1`

- `brainlayer_full`: `6/8`
- `brainlayer_no_forgetting`: `7/8`
- `structured_no_consolidation`: `6/8`

`gpt-4.1-mini`

- `brainlayer_full`: `6/8`
- `brainlayer_no_forgetting`: `5/8`
- `structured_no_consolidation`: `6/8`

This means forgetting now has real live evidence, but not uniform replication.

The clearest positive signal appears on:

- Gemini Flash
- Anthropic Haiku

The effect is flat or mixed on:

- Anthropic Sonnet
- OpenAI GPT-4.1
- OpenAI GPT-4.1 mini

So the honest conclusion is:

`Forgetting is promising, but tier-sensitive and not yet universal.`

### Consolidation

The redesigned consolidation pack produced a stronger result than the original `study-v2` ablations:

`gemini-2.5-flash`

- `brainlayer_full`: `5/8`
- `brainlayer_no_consolidation`: `3/8`
- `structured_no_consolidation`: `2/8`

`claude-haiku-4.5`

- `brainlayer_full`: `7/8`
- `brainlayer_no_consolidation`: `6/8`
- `structured_no_consolidation`: `6/8`

`claude-sonnet-4.5`

- `brainlayer_full`: `8/8`
- `brainlayer_no_consolidation`: `2/8`
- `structured_no_consolidation`: `2/8`

This is the strongest `study-v3` result.

It shows that consolidation can matter once the evaluation truly forces abstraction over longer gaps and weaker hints. The clearest version of that result is Anthropic Sonnet.

The Gemini result supports the same direction, but it remains more confounded by parse failures. Haiku shows a smaller positive signal.

So the honest consolidation conclusion is:

`Consolidation can matter, but only once the pack truly forces abstraction.`

## Final Defensible Claim

The strongest final claim supported by the full project is:

`BrainLayer's most replicated gains come from working state and autobiographical continuity. Forgetting appears promising but model-sensitive. Consolidation can produce real gains when the evaluation truly forces abstraction, but that effect is not yet as broadly replicated as the working-state and autobiographical results.`

This is a real result. It is narrower than the original ambition, but stronger because it is grounded in frozen runs, ablations, held-out packs, cross-provider checks, and targeted stress tests.

## What This Study Does Not Show

This study does **not** justify claiming:

- that full BrainLayer is universally better than simpler structured state
- that forgetting already helps robustly across all provider/model tiers
- that consolidation is broadly solved
- that the architecture is finished
- that agent cognition has been reduced to one settled design

Those would all overstate the evidence.

## Why The Result Still Matters

Even with those limits, the project now supports something important:

- the gains are not just benchmark scaffolding
- the gains are not just one-provider noise
- the most useful components are identifiable
- the weaker components are now clearly visible as open problems instead of hidden assumptions

That makes the project much more credible than a generic “memory helps agents” claim.

The stronger version of the result is:

`Agent improvement over time appears to require revisable cognitive state, not just note retrieval.`

But the study also shows that not every “brain-like” mechanism contributes equally under current task distributions.

## Current Bottom Line

If the question is “Do we have a publishable result?” the answer is yes, for a technical report, public research memo, workshop-style paper, or strong systems/agent writeup.

If the question is “Are we done proving the full BrainLayer vision?” the answer is no.

The finished part of the study is the evidence we already have:

- `working_state` matters
- `autobio` matters
- `forgetting` is promising
- `consolidation` can matter under the right pressure

The unfinished part is making the last two results broader, cleaner, and more universal.

## Recommended Next Step

Use this report as the final synthesis artifact for the current phase.

Then choose one of two paths:

1. publication path
   - turn this report into a paper/blog/report package with figures and tables

2. research continuation path
   - start a narrowly scoped `study-v4` whose only job is to make forgetting and consolidation replicate more broadly without changing the already-strong `working_state` and `autobio` findings

The important thing now is not to muddy the current result. The current result is strong enough to freeze and communicate.
