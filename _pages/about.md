---
layout: about
title: about
permalink: /
subtitle: "<strong>LLM post-training &amp; reinforcement learning</strong>"
header_email: kaili@cs.stanford.edu # shown under the name as "kaili [at] cs.stanford.edu"

profile:
  align: right
  video: stanford_2026_portrait.mp4 # living portrait (muted loop); poster is shown when video can't play
  poster: stanford_2026_portrait_poster.jpg
  video_alt: Kaili at Stanford, October 2026
  # image: prof_pic_3.jpg # static photo; used when `video` is removed
  image_circular: false # crops the image to make it circular
  more_info: >
    <div class="profile-quote">forward pass, backward pass, repeat.</div>
    <div class="profile-where">Stanford · Oct 2026</div>

intro: |
  I work on post-training and reinforcement learning for large language models. The parts I care about most are the ones that decide whether a model actually got better or only looks like it did: the <span class="kw">reward</span> that says what good means, the <span class="kw">data</span> the policy learns from, the <span class="kw">evaluation</span> that would catch it lying, and the <span class="kw">training loop</span> that holds the three together. At Apple I do this for LLM retrieval agents, from SFT through GRPO; I recently co-developed <a href="/projects/cogr/">CoGR</a>, which trains two LLMs to co-evolve against a single shared retrieval reward.

  I got here by way of dialogue. At Tsinghua I co-created the Chinese dialogue datasets people still train on: <a href="https://github.com/thu-coai/CrossWOZ">CrossWOZ</a> <a href="https://github.com/thu-coai/CrossWOZ"><img class="inline-badge" src="https://img.shields.io/github/stars/thu-coai/CrossWOZ?style=social" alt="GitHub stars for CrossWOZ" loading="lazy"></a>, <a href="https://github.com/thu-coai/KdConv">KdConv</a> <a href="https://github.com/thu-coai/KdConv"><img class="inline-badge" src="https://img.shields.io/github/stars/thu-coai/KdConv?style=social" alt="GitHub stars for KdConv" loading="lazy"></a>, and <a href="https://github.com/thu-coai/CDial-GPT">LCCC</a> <a href="https://github.com/thu-coai/CDial-GPT"><img class="inline-badge" src="https://img.shields.io/github/stars/thu-coai/CDial-GPT?style=social" alt="GitHub stars for LCCC" loading="lazy"></a>, which six years on still sees <a href="https://huggingface.co/datasets/thu-coai/lccc"><img class="inline-badge" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fthu-coai%2Flccc&query=%24.downloads&label=downloads%2Fmonth&logo=huggingface&logoColor=FFD21E&style=social" alt="Hugging Face monthly downloads for LCCC" loading="lazy"></a> on Hugging Face; at Stanford I built retrieval systems and tried model-based RL on dialogue policy for the first time; at ByteDance AI Lab I built and deployed fake-news and harmful-content detectors, from the annotation guidelines behind 2M+ human labels to automated retraining; at Microsoft AI I built the evaluation and alignment loop behind a retrieval-augmented Copilot. Each stop was a different piece of the same loop: the data a model learns from, the reward and training that make it better, and the check that it actually did.

  <div class="now-box" id="now">
    <strong>Thinking about:</strong> rewards that come from the real task rather than a proxy of it, how far an experiment loop can be handed to an agent before you stop trusting the result, and what an assistant has to understand about a person before it can take real weight off their mind, not just off their calendar. Chasing the same questions? <a href="mailto:kaili@cs.stanford.edu">Say hi</a>.
  </div>
selected_papers: false # includes a list of papers marked as "selected={true}"
social: true # includes social icons at the bottom of the page

announcements:
  enabled: false # includes a list of news items
  scrollable: true # adds a vertical scroll bar if there are more than 3 news items
  limit: 3 # leave blank to include all the news in the `_news` folder

latest_posts:
  enabled: false
  scrollable: true # adds a vertical scroll bar if there are more than 3 new posts items
  limit: 3 # leave blank to include all the blog posts
---

<ol class="journey" id="journey">
  <li>
    <div class="journey-where"><img class="journey-logo" src="/assets/img/logos/tsinghua.png" alt="" loading="lazy">Tsinghua</div>
    <div class="journey-when">2016 – 2020</div>
    <ul class="journey-list">
      <li><a href="/projects/dialogue-datasets/">CrossWOZ, KdConv, LCCC, CDial-GPT</a>: dialogue datasets and models people still use, with <a href="https://coai.cs.tsinghua.edu.cn/hml/">Minlie Huang</a></li>
      <li>3,000+ GitHub stars; NLPCC 2020 Best Student Paper</li>
    </ul>
  </li>
  <li>
    <div class="journey-where"><img class="journey-logo" src="/assets/img/logos/stanford.png" alt="" loading="lazy">Stanford</div>
    <div class="journey-when">2019 · 2021 – 2023</div>
    <ul class="journey-list">
      <li><a href="/projects/colbert-serve/">ColBERT-Serve</a> with <a href="https://people.eecs.berkeley.edu/~matei/">Matei Zaharia</a> (Databricks co-founder) and <a href="https://web.stanford.edu/~cgpotts/">Christopher Potts</a></li>
      <li><a href="/projects/rl-dialogue/">RL for dialogue</a> with <a href="https://scholar.google.com/citations?user=i38QlUwAAAAJ">Tengyu Ma</a>; CS224N TA with <a href="https://en.wikipedia.org/wiki/Christopher_D._Manning">Christopher Manning</a>; wrote the <a href="/assets/pdf/transformer_notes.pdf">Transformer notes</a></li>
    </ul>
  </li>
  <li>
    <div class="journey-where"><img class="journey-logo" src="/assets/img/logos/bytedance.svg" alt="" loading="lazy">ByteDance AI Lab</div>
    <div class="journey-when">2020 – 2021</div>
    <ul class="journey-list">
      <li>Fake-news and harmful-content detection at scale, with <a href="https://scholar.google.com/citations?user=nTl5mSwAAAAJ">Hang Li</a></li>
      <li>2M+ human labels with guidelines I iterated alongside the labeling team; automated retraining and rollout</li>
    </ul>
  </li>
  <li>
    <div class="journey-where"><img class="journey-logo" src="/assets/img/logos/microsoft.png" alt="" loading="lazy">Microsoft AI</div>
    <div class="journey-when">2023 – 2026</div>
    <ul class="journey-list">
      <li><a href="/projects/llm-evaluation/">Evaluation and alignment loop</a> for Copilot; co-authored <a href="https://arxiv.org/abs/2502.05497">DeepThink</a> on preference alignment</li>
      <li>Post-training for multimodal foundation models</li>
    </ul>
  </li>
  <li>
    <div class="journey-where"><img class="journey-logo" src="/assets/img/logos/apple.png" alt="" loading="lazy">Apple</div>
    <div class="journey-when">2026 –</div>
    <ul class="journey-list">
      <li>LLM post-training and RL: reward design and policy optimization</li>
      <li><a href="/projects/cogr/">CoGR</a>: query- and item-side LLMs that co-evolve through SFT and alternating GRPO against a shared retrieval reward</li>
    </ul>
  </li>
</ol>

<!-- intro -->

<div class="home-section">
<h2 class="section-title" id="ideas">ideas I keep returning to</h2>

<div class="idea-cards">
  <div class="idea-card">
    <div class="idea-title"><a href="/projects/autoresearch/">Agents that run their own experiments</a></div>
    <div class="idea-body">An agent proposes, trains, evaluates, and decides what to try next, inside rules a person sets.</div>
    <details class="idea-more">
      <summary>what it taught me about agency</summary>
      <p>Today's agents are good at pushing forward. What I want from the next generation is the judgment to stop: to step back and ask whether the whole approach is right, to change the architecture rather than the next line, and to reason from first principles: the goal is not to solve every problem it is handed, nor to grind on a particular one, but to keep the capacity to ask which problem is the right one. Most of my rule-setting in the experiment loop is an attempt to supply that judgment from the outside.</p>
    </details>
  </div>
  <div class="idea-card">
    <div class="idea-title">A second brain for everyone</div>
    <div class="idea-body">AI moves from a chatbot you visit to something you carry. It takes the tedious work off your mind, so you have the headspace to think and decide better.</div>
    <details class="idea-more">
      <summary>the longer bet</summary>
      <p>This is a long bet, measured in decades rather than product cycles. The hard parts are three: model capability, latency low enough that the system feels like part of you rather than a tool you wait on, and the interaction itself, how such a thing is built and wired into a life. It does the arithmetic, looks things up seamlessly as the conversation happens, remembers what you would otherwise have to memorize, helps you plan your life without the overhead, and clears away the tedious work that fills a day. You still do the thinking and make the decisions. It just makes sure nothing trivial is in the way. The measure of such a system is not how smart it sounds but how much clearer the person thinks, and how much better they decide, with it than without it. Underneath all of it is a simple belief: AI should serve people. Being free to do what you love and create what you want to create is one of the most valuable things a person could have. A system like this is worth building if it gives more people more of that freedom.</p>
    </details>
  </div>
</div>
</div>

<div class="home-section">
<h2 id="projects"><a href="/projects/" style="color: inherit">selected projects</a></h2>
{% assign featured = site.projects | where: "featured", true | sort: "importance" %}
<div class="projects">
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in featured %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
</div>
</div>
