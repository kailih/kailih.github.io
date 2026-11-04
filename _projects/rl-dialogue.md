---
layout: page
title: "Model-Based RL for Dialogue Policy"
description: Built SLBO and vanilla policy-gradient agents for a movie-ticket booking task, a chatbot environment and user-simulator sampling pipeline to train them, and a harness for comparing model-based deep RL algorithms on dialogue policy. Stanford UGVR with Tengyu Ma.
img: assets/img/projects/rl-dialogue.png
importance: 6
category: Research
permalink: /projects/rl-dialogue/
---

<div class="project-meta">
  <span>2019 · Stanford, advisor Tengyu Ma</span>
  <span>UGVR scholar</span>
  <span>RL · environments · dialogue</span>
</div>

**The problem.** Dialogue policy is a natural RL problem (long horizon, sparse reward at task completion), but real users are expensive and model-free methods are sample hungry. Model-based RL promises to learn a dynamics model of the user and plan against it.

**What I built.**

- A **chatbot environment** for the movie-ticket booking task with a user simulator, so a policy could be trained end to end against simulated users.
- A **vanilla policy gradient (VPG)** agent as the model-free baseline.
- A **Stochastic Lower Bound Optimization (SLBO)** agent, a model-based algorithm, adapted to dialogue state and action spaces.
- A **pipeline** that samples trajectories from the simulator, fits the dynamics model, and trains policies under different model-based deep RL algorithms, so they can be compared on the same budget of simulator interactions.

**Context.** Research project as a Stanford UGVR scholar (up to 18 students from China admitted per year).
