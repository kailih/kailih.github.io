---
layout: page
title: "LLM Evaluation & Alignment"
description: Three threads from Microsoft AI. Rubric-based LLM-judge evaluation with human-in-the-loop metrics for Copilot's retrieval-augmented generation; post-training for multimodal retrieval models (Qwen-VL, CLIP-style); and an LLM teacher distilled into a lightweight E5 retriever that shipped after A/B testing.
img: assets/img/projects/llm-evaluation.png
importance: 5
category: Research
permalink: /projects/llm-evaluation/
---

<div class="project-meta">
  <span>2023–2026 · Microsoft AI</span>
  <span>evaluation · RAG · multimodal post-training · distillation</span>
</div>

<div class="project-stats">
  <div class="stat"><div class="stat-value">+15.6%</div><div class="stat-label">Recall@10 on Copilot RAG retrieval</div></div>
  <div class="stat"><div class="stat-value">+6%</div><div class="stat-label">overall answer quality</div></div>
  <div class="stat"><div class="stat-value">+28–34%</div><div class="stat-label">AUC from distilled student, shipped after A/B</div></div>
</div>

### 1. Evaluation and alignment loop for Copilot RAG

**The problem.** A RAG-based Copilot can go wrong at every link in the chain. The generator can produce a wrong or ungrounded answer even when it is handed the right documents. The retriever can miss the right documents, or surface the wrong ones with confidence. And the source documents themselves can be wrong or out of date, so the fix is to update the knowledge, not the model. Each of these fails quietly and looks like the others from the outside. You need an evaluator that can tell them apart and run on every change, and you need humans in the loop to tell you when the evaluator itself is wrong.

**What I built.**

- **Rubric-based LLM-judge evaluation.** GPT-based reward-proxy evaluation with chain-of-thought prompting, decomposed into modular rubrics so groundedness, instruction following, and hallucination are each scored on their own and can gate a release.
- **Human-in-the-loop metrics and eval sets.** Curated high-quality, human-reviewed evaluation sets, defined the metrics that human review and the judge share, and wired the core metrics into real-time dashboards so regressions showed up the same day. After launch, user feedback from production fed the same loop and drove many further iterations of both the system and the evaluation sets.
- **Retrieval and product improvements.** Using that evaluation loop, improved the retrieval algorithm (+15.6% Recall@10, +6% overall answer quality) and contributed production LLM features: multi-turn conversation, user-intent rewriting, and frustration detection.

<div class="project-figure">
<svg viewBox="0 0 760 290" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" font-size="14" fill="currentColor">
  <defs>
    <marker id="arr4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>
  <!-- eval sets define the rubrics -->
  <rect x="200" y="12" width="190" height="54" rx="8" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="295" y="34" text-anchor="middle" font-weight="600">Human-reviewed eval sets</text>
  <text x="295" y="52" text-anchor="middle" font-size="12" opacity="0.75">define and calibrate the rubrics</text>
  <line x1="295" y1="66" x2="295" y2="100" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr4)"/>
  <!-- main row -->
  <rect x="20" y="100" width="150" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="95" y="126" text-anchor="middle" font-weight="600">RAG candidate</text>
  <text x="95" y="146" text-anchor="middle" font-size="12" opacity="0.75">retriever / prompt change</text>
  <line x1="170" y1="130" x2="220" y2="130" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr4)"/>
  <rect x="220" y="100" width="150" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="295" y="126" text-anchor="middle" font-weight="600">Rubric judges</text>
  <text x="295" y="146" text-anchor="middle" font-size="12" opacity="0.75">CoT reward proxies</text>
  <line x1="370" y1="130" x2="420" y2="130" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr4)"/>
  <rect x="420" y="100" width="150" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="495" y="126" text-anchor="middle" font-weight="600">Dashboards</text>
  <text x="495" y="146" text-anchor="middle" font-size="12" opacity="0.75">gate the release</text>
  <line x1="570" y1="130" x2="620" y2="130" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr4)"/>
  <rect x="620" y="100" width="120" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="680" y="126" text-anchor="middle" font-weight="600">Ship</text>
  <text x="680" y="146" text-anchor="middle" font-size="12" opacity="0.75">to users</text>
  <!-- user feedback -->
  <rect x="300" y="220" width="220" height="54" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="410" y="242" text-anchor="middle" font-weight="600">User feedback</text>
  <text x="410" y="260" text-anchor="middle" font-size="12" opacity="0.75">from production</text>
  <path d="M680,160 C680,262 560,262 520,262" fill="none" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr4)"/>
  <path d="M300,247 C180,247 95,230 95,160" fill="none" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr4)"/>
  <text x="190" y="205" text-anchor="middle" font-size="12" opacity="0.75">iterate the system</text>
  <path d="M520,232 C640,232 756,232 756,150 L756,60 C756,39 740,39 720,39 L390,39" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arr4)"/>
  <text x="560" y="30" text-anchor="middle" font-size="12" opacity="0.75">refresh the eval sets</text>
</svg>
<div class="caption">Human-reviewed eval sets set up the rubric judges; every change is judged and gated before it ships; user feedback from production then iterates both the system and the eval sets.</div>
</div>

### 2. Post-training for multimodal retrieval

Built distributed PyTorch fine-tuning and evaluation pipelines for Qwen-VL and CLIP-style models (CLIP, SigLIP, Perception Encoder) across five multimodal retrieval and reranking tasks, and led the post-training that improved AUC by up to 12%.

### 3. Teacher-to-student distillation for text retrieval

Post-trained an LLM teacher for text-to-text retrieval and distilled it into a lightweight E5 student: +28–34% AUC over the production baseline, and −6.2% defect rate in production A/B tests, after which the student shipped.

**What I did.** The multimodal retrieval work and the distillation work were my projects end to end. On Copilot RAG, I owned the evaluation pipeline, from judge design to human-in-the-loop metrics and dashboards, while other teammates developed the product features.
