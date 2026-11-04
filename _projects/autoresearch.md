---
layout: page
title: "Autoresearch: an agent that runs the experiment loop"
description: A generic workflow in which an LLM agent proposes, trains, evaluates, and accepts or rejects experiments on its own, inside rules a person sets. I built one instance of it for model compression; after manual validation of the first iterations, it ran a multi-day sequence of experiments without further input and delivered a materially smaller model within the performance tolerance.
img: assets/img/projects/autoresearch.png
importance: 1
category: Research
featured: true
permalink: /projects/autoresearch/
---

<div class="project-meta">
  <span>2026</span>
  <span>automated experimentation · agents · model compression</span>
</div>

<div class="project-stats">
  <div class="stat"><div class="stat-value">unattended</div><div class="stat-label">multi-day run after initial validation</div></div>
  <div class="stat"><div class="stat-value">performance-gated</div><div class="stat-label">every candidate accepted by rule, not judgment</div></div>
  <div class="stat"><div class="stat-value">reproducible</div><div class="stat-label">one versioned manifest per experiment</div></div>
</div>

**The pattern.** Autoresearch is a simple idea that many people are converging on: hand the experiment loop to an LLM agent and keep the human on the rules. The loop is generic (propose a change, run it, read the results, decide, repeat), so the same scaffold applies to any problem where each candidate needs a training run, an evaluation, and a judgment call. A person can run a handful of these a day and loses context between them; an agent can run them around the clock and keep every result on the table.

**The instance.** I built one for model compression, where the search space is changes to features, architecture, and embeddings, and the acceptance criterion is a fixed performance tolerance.

**What the workflow does.** The agent owns the loop; the human owns the rules.

- **Search constraints and rules.** I defined the space of allowed changes (feature, architecture, and embedding changes), how they may be alternated and combined, and what counts as an improvement. The agent proposes within that space.
- **Experiment manifests.** Each candidate becomes a versioned manifest that fully specifies the run, so results are reproducible and comparable.
- **Concurrent training and inference jobs.** The agent launches and tracks several runs at once and collects their artifacts.
- **Analysis of logs and evaluation results.** It reads training logs and evaluation output, diagnoses failed or degenerate runs, and summarizes what moved.
- **Performance-gated acceptance.** A candidate is accepted only if performance stays within a fixed tolerance of the baseline. The gate is a rule, not a judgment call, so the agent cannot talk itself into a bad trade.
- **Combine, then retest.** When several candidates pass the gate in the same round, the agent combines them and runs the combination. If the combination also passes, it becomes the new baseline. If it does not, the agent keeps the single change with the largest compression gain and continues from there.

<div class="project-figure">
<svg viewBox="0 0 760 300" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" font-size="14" fill="currentColor">
  <defs>
    <marker id="arr5" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>
  <!-- human box -->
  <rect x="0" y="12" width="240" height="84" rx="8" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="120" y="40" text-anchor="middle" font-weight="600">Human: rules</text>
  <text x="120" y="60" text-anchor="middle" font-size="12" opacity="0.75">search space · constraints</text>
  <text x="120" y="76" text-anchor="middle" font-size="12" opacity="0.75">performance gate</text>
  <line x1="120" y1="96" x2="120" y2="130" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr5)"/>
  <!-- loop boxes -->
  <rect x="20" y="130" width="200" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="120" y="155" text-anchor="middle" font-weight="600">Propose</text>
  <text x="120" y="175" text-anchor="middle" font-size="12" opacity="0.75">next change → manifest</text>
  <line x1="220" y1="160" x2="280" y2="160" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr5)"/>
  <rect x="280" y="130" width="200" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="380" y="155" text-anchor="middle" font-weight="600">Run</text>
  <text x="380" y="175" text-anchor="middle" font-size="12" opacity="0.75">concurrent train + inference jobs</text>
  <line x1="480" y1="160" x2="540" y2="160" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr5)"/>
  <rect x="540" y="130" width="200" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="640" y="155" text-anchor="middle" font-weight="600">Analyze</text>
  <text x="640" y="175" text-anchor="middle" font-size="12" opacity="0.75">logs · eval results · diagnosis</text>
  <!-- gate -->
  <rect x="290" y="230" width="250" height="54" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="415" y="252" text-anchor="middle" font-weight="600">Performance gate</text>
  <text x="415" y="272" text-anchor="middle" font-size="12" opacity="0.75">accept if Δperformance within tolerance</text>
  <path d="M640,190 C640,257 540,257 540,257" fill="none" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr5)"/>
  <!-- combine -->
  <rect x="10" y="230" width="250" height="54" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="135" y="252" text-anchor="middle" font-weight="600">Combine survivors, retest</text>
  <text x="135" y="272" text-anchor="middle" font-size="12" opacity="0.75">else keep the single biggest win</text>
  <line x1="290" y1="257" x2="260" y2="257" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr5)"/>
  <line x1="135" y1="230" x2="135" y2="190" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr5)"/>
  <text x="150" y="214" text-anchor="start" font-size="12" opacity="0.75">new baseline, next round</text>
</svg>
<div class="caption">The human sets the rules once; the agent runs propose → run → analyze → gate → combine for as long as the budget allows.</div>
</div>

**What happened.** After I validated the first few iterations by hand, the workflow ran on its own for days, chaining dozens of experiments. The accepted sequence of changes produced a materially smaller model while staying within the performance tolerance.

**What I learned.** The agent is only as good as the rules it is handed: a precise search space and a hard acceptance gate mattered more than any prompt. Reproducible manifests made the run auditable after the fact, which is what let me trust the result. The scaffold itself is problem-agnostic; swapping the search space and the gate is enough to point it at a different question.

The other lesson is that none of this needed a framework. I started out assuming I would have to adopt someone else's autoresearch codebase. Instead I built the loop together with a coding agent, and after a few rounds of that I understood exactly what the loop needed and where to hand it over. Autoresearch is something anyone can build for their own problem; the idea is simple, and the work is in the rules.
