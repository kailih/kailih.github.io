---
layout: page
title: "Chinese Dialogue Corpora + DSTC9"
description: Three widely used open datasets for Chinese dialogue (CrossWOZ, KdConv, LCCC), each shipped with benchmark models and, for CrossWOZ, a user simulator. Together they have 3,000+ GitHub stars. I also co-organized the DSTC9 multi-domain task-completion track built on them.
img: assets/img/projects/dialogue-datasets.png
importance: 3
category: Research
permalink: /projects/dialogue-datasets/
---

<div class="project-meta">
  <span>2018–2021 · Tsinghua CoAI, Microsoft Research</span>
  <a href="https://aclanthology.org/2020.tacl-1.19">CrossWOZ (TACL)</a>
  <a href="https://aclanthology.org/2020.acl-main.635">KdConv (ACL)</a>
  <a href="https://arxiv.org/abs/2008.03946">LCCC (NLPCC, best student paper)</a>
  <a href="https://ieeexplore.ieee.org/document/10595468">DSTC9 overview (TASLP)</a>
  <a href="https://huggingface.co/datasets/GEM/CrossWOZ">CrossWOZ on Hugging Face</a>
  <a href="https://huggingface.co/datasets/thu-coai/lccc">LCCC on Hugging Face</a>
  <a href="https://huggingface.co/datasets/thu-coai/kdconv">KdConv on Hugging Face</a>
  <span>data · environments · benchmarks</span>
</div>

<div class="project-stats">
  <div class="stat"><div class="stat-value">3,200+</div><div class="stat-label">GitHub stars across the three repos</div></div>
  <div class="stat"><div class="stat-value">12M</div><div class="stat-label">cleaned dialogues in LCCC-large</div></div>
  <div class="stat"><div class="stat-value">~1K / month</div><div class="stat-label">Hugging Face downloads of LCCC, six years after release</div></div>
</div>

Good data is an environment. Each of these corpora was built so that other people could train against it, evaluate on it, and compare. Six years on they are still in use: as of late 2026 the Hugging Face copies of LCCC, KdConv, and CrossWOZ are each still downloaded hundreds to a thousand times a month.

- **[CrossWOZ](https://github.com/thu-coai/CrossWOZ)** (TACL 2020) <a href="https://github.com/thu-coai/CrossWOZ"><img class="inline-badge" src="https://img.shields.io/github/stars/thu-coai/CrossWOZ?style=social" alt="GitHub stars for CrossWOZ" loading="lazy"></a> <a href="https://huggingface.co/datasets/GEM/CrossWOZ"><img class="inline-badge" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FGEM%2FCrossWOZ&query=%24.downloads&label=downloads%2Fmonth&logo=huggingface&logoColor=FFD21E&style=social" alt="Hugging Face monthly downloads for CrossWOZ (GEM mirror)" loading="lazy"></a>: the first large-scale Chinese cross-domain Wizard-of-Oz task-oriented dataset. 6K dialogues, 102K utterances, 5 domains (hotel, restaurant, attraction, metro, taxi), dialogue state and dialogue act annotation on both user and system side. About 60% of dialogues have cross-domain goals. Ships with a **user simulator** and pipelined benchmark models, so it doubles as an RL environment for dialogue policy.
- **[KdConv](https://github.com/thu-coai/KdConv)** (ACL 2020) <a href="https://github.com/thu-coai/KdConv"><img class="inline-badge" src="https://img.shields.io/github/stars/thu-coai/KdConv?style=social" alt="GitHub stars for KdConv" loading="lazy"></a> <a href="https://huggingface.co/datasets/thu-coai/kdconv"><img class="inline-badge" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fthu-coai%2Fkdconv&query=%24.downloads&label=downloads%2Fmonth&logo=huggingface&logoColor=FFD21E&style=social" alt="Hugging Face monthly downloads for KdConv" loading="lazy"></a>: a multi-domain knowledge-driven dialogue dataset grounding 4.5K conversations (86K utterances, 19 turns on average) in knowledge graphs across film, music, and travel. Comes with benchmark models that show how much background knowledge helps.
- **[LCCC and CDial-GPT](https://github.com/thu-coai/CDial-GPT)** (NLPCC 2020, best student paper) <a href="https://github.com/thu-coai/CDial-GPT"><img class="inline-badge" src="https://img.shields.io/github/stars/thu-coai/CDial-GPT?style=social" alt="GitHub stars for CDial-GPT" loading="lazy"></a> <a href="https://huggingface.co/datasets/thu-coai/lccc"><img class="inline-badge" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fthu-coai%2Flccc&query=%24.downloads&label=downloads%2Fmonth&logo=huggingface&logoColor=FFD21E&style=social" alt="Hugging Face monthly downloads for LCCC" loading="lazy"></a>: a 6.8M / 12M dialogue corpus produced by a rigorous cleaning pipeline (rules plus a classifier trained on 110K manually annotated pairs), with pre-trained dialogue models released on [Hugging Face](https://huggingface.co/thu-coai/CDial-GPT_LCCC-large). This is the most starred of the three.
- **[DSTC9 Track 2](https://www.microsoft.com/en-us/research/publication/multi-domain-task-oriented-dialog-challenge-ii-at-dstc9/)** (AAAI 2021 workshop, TASLP 2024 overview): with Microsoft Research and Tsinghua, I co-organized the multi-domain task-completion challenge, which added an end-to-end task-completion task and a cross-lingual dialogue state tracking task built on CrossWOZ and MultiWOZ.

**What I did.** Worked across the whole pipeline with the team: collected the initial seed data from scratch, took part in the human data collection, and helped clean the corpora and train the benchmark models. Second author on CrossWOZ, co-author on KdConv and LCCC, and co-organizer of DSTC9 Track 2 with Microsoft Research. LCCC received the NLPCC 2020 Best Student Paper Award.
