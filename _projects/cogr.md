---
layout: page
title: "CoGR: Co-Evolving Generative Retrieval"
description: Two LLMs, one for queries and one for items, learn to write matching keyword representations. After SFT they are trained with alternating GRPO updates against a single shared retrieval objective, with a counterfactual marginal reward for item-level credit assignment. 10.9–36.1% relative F1 gains over the strongest baseline.
img: assets/img/projects/cogr.png
importance: 0
category: Research
featured: true
permalink: /projects/cogr/
---

<div class="project-meta">
  <span>2026 · Apple</span>
  <a href="https://arxiv.org/abs/2609.00638">arXiv paper</a>
  <a href="https://arxiv.org/pdf/2609.00638">PDF</a>
  <span>LLM post-training · GRPO · verl · Qwen3-4B · retrieval</span>
</div>

**The problem.** Dense retrievers compare a query embedding with an item embedding. That works, but the representation is opaque and the two towers are trained once and frozen relative to each other. We wanted retrieval that an LLM can reason about, and that keeps improving on both sides.

<div class="project-stats">
  <div class="stat"><div class="stat-value">+10.9–36.1%</div><div class="stat-label">relative F1 over the strongest baseline</div></div>
  <div class="stat"><div class="stat-value">SFT → GRPO</div><div class="stat-label">alternating query-side and item-side updates</div></div>
  <div class="stat"><div class="stat-value">counterfactual</div><div class="stat-label">marginal reward for item-level credit</div></div>
</div>

**The idea.** CoGR trains an LLM to construct a compact keyword representation for a query, and another to construct one for an item. Supervised fine-tuning first establishes a shared keyword space. Then alternating GRPO updates optimize the query-side and item-side generators against the same retrieval objective. The query side is rewarded directly by retrieval F1. The item side gets a **counterfactual marginal reward**: how much does the query-side retrieval F1 change when this item's keywords are swapped in, holding everything else fixed? That gives each item its own credit signal instead of a shared, noisy one. Each side is trained against the current state of the other, so the two representations evolve toward each other instead of apart.

<div class="project-figure">
<svg viewBox="0 0 760 300" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" font-size="14" fill="currentColor">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>
  <!-- query side -->
  <rect x="20" y="40" width="200" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="120" y="64" text-anchor="middle" font-weight="600">Query generator (LLM)</text>
  <text x="120" y="86" text-anchor="middle" font-size="12" opacity="0.75">query → keywords</text>
  <!-- item side -->
  <rect x="20" y="200" width="200" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="120" y="224" text-anchor="middle" font-weight="600">Item generator (LLM)</text>
  <text x="120" y="246" text-anchor="middle" font-size="12" opacity="0.75">item → keywords</text>
  <!-- keyword space -->
  <rect x="300" y="110" width="180" height="80" rx="40" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="390" y="144" text-anchor="middle" font-weight="600">Shared keyword</text>
  <text x="390" y="164" text-anchor="middle" font-weight="600">space</text>
  <!-- reward -->
  <rect x="560" y="110" width="180" height="80" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="650" y="140" text-anchor="middle" font-weight="600">Retrieval reward</text>
  <text x="650" y="160" text-anchor="middle" font-size="12" opacity="0.75">does q match the right item?</text>
  <text x="650" y="176" text-anchor="middle" font-size="12" opacity="0.75">(one objective, both sides)</text>
  <!-- arrows into space -->
  <line x1="220" y1="70" x2="300" y2="130" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="220" y1="230" x2="300" y2="170" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr)"/>
  <line x1="480" y1="150" x2="560" y2="150" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- RL feedback arrows -->
  <path d="M650,110 C650,20 150,20 120,40" fill="none" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="400" y="26" text-anchor="middle" font-size="12" opacity="0.75">GRPO on retrieval F1, odd rounds</text>
  <path d="M650,190 C650,285 150,285 120,260" fill="none" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="400" y="290" text-anchor="middle" font-size="12" opacity="0.75">GRPO on counterfactual marginal reward, even rounds</text>
</svg>
<div class="caption">The two generators share one reward and are updated in alternation, so each is always optimized against the other's current behavior.</div>
</div>

**What I did.** Co-developed the framework and mentored the intern who led the project.

**Why I like it.** The reward is the real retrieval task, not a proxy for it. And because the two sides are trained in alternation, each one keeps adapting to what the other has just learned. That makes the setup powerful, and also one you have to watch closely: the evaluation has to tell you whether the two sides are converging on a shared language or collapsing into a degenerate one.
