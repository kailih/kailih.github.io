---
layout: page
title: "ColBERT-Serve"
description: A serving system that memory-maps the ColBERT index, uses a sparse first stage (SPLADEv2) to pick candidates, and re-ranks them with ColBERT under a hybrid score. Cuts RAM by 90% and lets late-interaction retrieval run on inexpensive servers under many concurrent queries. ECIR 2025, with Matei Zaharia (Databricks co-founder), Christopher Potts, et al.
img: assets/img/projects/colbert-serve.png
importance: 2
category: Research
featured: true
permalink: /projects/colbert-serve/
github: https://github.com/stanford-futuredata/colbert-serve
---

<div class="project-meta">
  <span>2023–2025 · started at Stanford, published from Microsoft</span>
  <a href="https://arxiv.org/abs/2504.14903">ECIR 2025 paper</a>
  <a href="https://github.com/stanford-futuredata/colbert-serve">Code</a>
  <a href="https://github.com/stanford-futuredata/colbert-serve"><img class="inline-badge" src="https://img.shields.io/github/stars/stanford-futuredata/colbert-serve?style=social" alt="GitHub stars for ColBERT-Serve" loading="lazy"></a>
  <span>retrieval · serving · systems</span>
</div>

<div class="project-stats">
  <div class="stat"><div class="stat-value">&gt;90%</div><div class="stat-label">less RAM for the ColBERT index</div></div>
  <div class="stat"><div class="stat-value">multi-stage</div><div class="stat-label">SPLADE candidates, ColBERT re-rank with hybrid scoring</div></div>
  <div class="stat"><div class="stat-value">ECIR 2025</div><div class="stat-label">first author</div></div>
</div>

**The problem.** ColBERT is one of the best retrievers available, but its index stores a vector per token, so it does not fit in memory on a cheap server. Most deployments either pay for RAM or give up on late interaction.

**The idea.** Treat the index like a database, not a tensor. ColBERT-Serve memory-maps the compressed ColBERTv2 embeddings from disk and relies on the OS page cache, so only the pages a query touches get loaded. Memory-mapping alone makes search about 2× slower because of page misses, so a sparse first stage (SPLADEv2 running on the PISA engine) picks the top-200 candidates and ColBERT only scores those. The final ranking uses a hybrid score, a z-normalized interpolation of the SPLADE and ColBERT scores, which on MS MARCO beats even full in-memory ColBERTv2. The result serves several queries per second on a machine with a few GB of RAM, at about a quarter of the cost.

<div class="project-figure">
<svg viewBox="0 0 760 250" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" font-size="14" fill="currentColor">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
    </marker>
  </defs>
  <rect x="10" y="70" width="110" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="65" y="105" text-anchor="middle" font-weight="600">Queries</text>
  <line x1="120" y1="100" x2="165" y2="100" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr2)"/>
  <rect x="165" y="55" width="180" height="90" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="255" y="85" text-anchor="middle" font-weight="600">Stage 1: SPLADEv2</text>
  <text x="255" y="105" text-anchor="middle" font-size="12" opacity="0.75">sparse retrieval on PISA</text>
  <text x="255" y="123" text-anchor="middle" font-size="12" opacity="0.75">full corpus → top-200</text>
  <line x1="345" y1="100" x2="380" y2="100" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr2)"/>
  <rect x="380" y="55" width="200" height="90" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="480" y="85" text-anchor="middle" font-weight="600">Stage 2: ColBERT re-rank</text>
  <text x="480" y="105" text-anchor="middle" font-size="12" opacity="0.75">late interaction on top-200</text>
  <text x="480" y="123" text-anchor="middle" font-size="12" opacity="0.75">hybrid score: SPLADE + ColBERT</text>
  <line x1="580" y1="100" x2="630" y2="100" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr2)"/>
  <rect x="630" y="70" width="120" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="690" y="105" text-anchor="middle" font-weight="600">Results</text>
  <!-- index -->
  <rect x="330" y="180" width="290" height="58" rx="8" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="475" y="200" text-anchor="middle" font-weight="600">ColBERT index on disk (mmap)</text>
  <text x="475" y="216" text-anchor="middle" font-size="12" opacity="0.75">only touched pages enter RAM</text>
  <text x="475" y="230" text-anchor="middle" font-size="12" opacity="0.75">via the OS page cache</text>
  <line x1="480" y1="145" x2="480" y2="180" stroke="currentColor" stroke-width="1.5" marker-end="url(#arr2)" marker-start="url(#arr2)"/>
</svg>
<div class="caption">SPLADEv2 picks the candidates; ColBERT scores only those against a memory-mapped index, and the two scores are blended.</div>
</div>

**Try it.** The repository includes the serving code and instructions to reproduce the memory and latency results on your own index.
