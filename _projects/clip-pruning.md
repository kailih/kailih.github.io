---
layout: page
title: "Structured Pruning for CLIP"
description: Extended structured pruning to a multimodal dual-encoder for the first time, removing 40% of CLIP's parameters at a 1% accuracy cost. Built the sparse PyTorch training pipeline, decomposed Block Movement Pruning into modular steps, and added auxiliary losses to balance the two modalities.
img: assets/img/projects/clip-pruning.png
importance: 4
category: Research
permalink: /projects/clip-pruning/
---

<div class="project-meta">
  <span>2022 · Microsoft AI (internship)</span>
  <span>model compression · multimodal · efficiency</span>
</div>

<div class="project-stats">
  <div class="stat"><div class="stat-value">−40%</div><div class="stat-label">parameters removed from CLIP</div></div>
  <div class="stat"><div class="stat-value">−1%</div><div class="stat-label">accuracy, the only cost</div></div>
  <div class="stat"><div class="stat-value">first</div><div class="stat-label">structured pruning applied to a dual-encoder</div></div>
</div>

**The problem.** Structured pruning (removing whole attention heads and FFN channels, so the model actually runs faster, not just smaller) was well studied for BERT-style encoders. Nobody had applied it to a contrastive dual-encoder like CLIP, where pruning one tower silently breaks its alignment with the other.

**The idea.** Prune both towers jointly under a shared budget, and treat the contrastive alignment as something to protect, not just the per-tower accuracy. I built a sparse training pipeline in PyTorch, decomposed Block Movement Pruning into modular steps so each could be scheduled and ablated, and introduced auxiliary losses that stabilize training and balance the two modalities while capacity is being removed.

<div class="project-figure">
<svg viewBox="0 0 760 230" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" font-size="14" fill="currentColor">
  <defs>
    <marker id="arr3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>
  <rect x="20" y="30" width="180" height="70" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="110" y="60" text-anchor="middle" font-weight="600">Image encoder</text>
  <text x="110" y="82" text-anchor="middle" font-size="12" opacity="0.75">heads + FFN channels masked</text>
  <rect x="20" y="130" width="180" height="70" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="110" y="160" text-anchor="middle" font-weight="600">Text encoder</text>
  <text x="110" y="182" text-anchor="middle" font-size="12" opacity="0.75">heads + FFN channels masked</text>
  <line x1="200" y1="65" x2="290" y2="105" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr3)"/>
  <line x1="200" y1="165" x2="290" y2="125" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr3)"/>
  <rect x="290" y="80" width="190" height="70" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="385" y="108" text-anchor="middle" font-weight="600">Contrastive loss</text>
  <text x="385" y="130" text-anchor="middle" font-size="12" opacity="0.75">+ alignment auxiliary losses</text>
  <line x1="480" y1="115" x2="560" y2="115" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr3)"/>
  <rect x="560" y="80" width="180" height="70" rx="8" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="650" y="108" text-anchor="middle" font-weight="600">Shared sparsity budget</text>
  <text x="650" y="130" text-anchor="middle" font-size="12" opacity="0.75">prune → recover → repeat</text>
  <path d="M650,150 C650,215 110,230 110,200" fill="none" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr3)"/>
  <text x="400" y="220" text-anchor="middle" font-size="12" opacity="0.75">mask updates flow back to both towers</text>
</svg>
<div class="caption">Both towers are pruned under one budget, and the recovery schedule retrains on the contrastive objective so the towers stay aligned.</div>
</div>

**What I did.** This was my internship project, and I did it end to end over three months: designed the method, built the sparse training pipeline, and ran the ablations over pruning schedules and auxiliary losses.
