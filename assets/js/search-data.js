// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-projects",
          title: "projects",
          description: "Research I have done, from RL post-training and evaluation to retrieval and open datasets, plus the open-source repositories behind it. Teaching and coursework from Stanford at the bottom.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "Kaili&#39;s publications in reversed chronological order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "Kaili Huang&#39;s CV.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "nav-news",
          title: "news",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/news/";
          },
        },{id: "news-excited-to-share-that-our-paper-colbert-serve-efficient-multi-stage-memory-mapped-scoring-was-accepted-and-presented-at-the-47th-european-conference-on-information-retrieval-ecir-2025-the-work-introduces-a-memory-mapped-multi-stage-scoring-approach-that-makes-late-interaction-neural-retrieval-colbert-far-more-efficient-and-scalable-for-serving-llm-powered-retrieval-systems",
          title: 'Excited to share that our paper, ColBERT-serve: Efficient Multi-Stage Memory-Mapped Scoring, was accepted...',
          description: "",
          section: "News",},{id: "news-i-m-happy-to-share-that-i-joined-apple-as-a-senior-machine-learning-engineer-this-january-working-on-llm-post-training-rl-policy-optimization-and-alignment-evaluation-for-large-scale-retrieval-i-look-forward-to-pushing-the-boundaries-of-reasoning-driven-retrieval-and-alignment-read-more-on-linkedin",
          title: 'I’m happy to share that I joined Apple as a Senior Machine Learning...',
          description: "",
          section: "News",},{id: "news-excited-to-share-our-latest-work-it-takes-two-to-match-co-evolving-generative-retriever-with-reinforcement-learning-we-introduce-cogr-a-generative-retrieval-framework-in-which-llms-generate-matching-keywords-for-both-queries-and-items-and-the-two-sides-co-evolve-through-reinforcement-learning-check-out-my-posts-on-linkedin-and-x",
          title: 'Excited to share our latest work, It Takes Two to Match: Co-Evolving Generative...',
          description: "",
          section: "News",},{id: "projects-autoresearch-an-agent-that-runs-the-experiment-loop",
          title: 'Autoresearch: an agent that runs the experiment loop',
          description: "A generic workflow in which an LLM agent proposes, trains, evaluates, and accepts or rejects experiments on its own, inside rules a person sets. I built one instance of it for model compression; after manual validation of the first iterations, it ran a multi-day sequence of experiments without further input and delivered a materially smaller model within the performance tolerance.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/autoresearch/";
            },},{id: "projects-structured-pruning-for-clip",
          title: 'Structured Pruning for CLIP',
          description: "Extended structured pruning to a multimodal dual-encoder for the first time, removing 40% of CLIP&#39;s parameters at a 1% accuracy cost. Built the sparse PyTorch training pipeline, decomposed Block Movement Pruning into modular steps, and added auxiliary losses to balance the two modalities.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/clip-pruning/";
            },},{id: "projects-cogr-co-evolving-generative-retrieval",
          title: 'CoGR: Co-Evolving Generative Retrieval',
          description: "Two LLMs, one for queries and one for items, learn to write matching keyword representations. After SFT they are trained with alternating GRPO updates against a single shared retrieval objective, with a counterfactual marginal reward for item-level credit assignment. 10.9–36.1% relative F1 gains over the strongest baseline.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/cogr/";
            },},{id: "projects-colbert-serve",
          title: 'ColBERT-Serve',
          description: "A serving system that memory-maps the ColBERT index, uses a sparse first stage (SPLADEv2) to pick candidates, and re-ranks them with ColBERT under a hybrid score. Cuts RAM by 90% and lets late-interaction retrieval run on inexpensive servers under many concurrent queries. ECIR 2025, with Matei Zaharia (Databricks co-founder), Christopher Potts, et al.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/colbert-serve/";
            },},{id: "projects-engineering-effective-in-context-inputs-for-gpt-3-in-openqa",
          title: 'Engineering Effective In-Context Inputs for GPT-3 in OpenQA',
          description: "Designed and evaluated novel in-context learning strategies to improve GPT-3’s performance on OpenQA without access to gold passages. Explored lexical, syntactic, and semantic similarity-based example selection methods, and introduced reverse ordering to enhance contextual relevance. The semantic similarity + reverse order strategy achieved the best performance (F1=0.57), yielding a 5% improvement over the random baseline. Findings highlight the impact of example amount, quality, similarity, and ordering on large language model effectiveness.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/cs224u_22spr/";
            },},{id: "projects-optimizing-dialogue-history-encoding-for-multilingual-task-oriented-systems",
          title: 'Optimizing Dialogue History Encoding for Multilingual Task-Oriented Systems',
          description: "Proposed an efficient training strategy for multilingual virtual assistants by replacing natural language dialogue history with structured dialogue states. Built on the BiToD architecture to reduce reliance on weak natural language encoders and improve slot value extraction. The work investigated the impact of history length (number of previous turns) on model performance, identified diminishing returns, and explored few-shot learning for hard examples. It is designed to improve scalability and robustness in low-resource languages.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/cs224v_21fall/";
            },},{id: "projects-which-trick-works-best-in-gnn",
          title: 'Which Trick Works Best in GNN?',
          description: "In this blog, we reproduced the work of &quot;Residual Network and Embedding Usage - New Tricks of Node Classification with Graph Convolutional Networks.&quot; and deliver a thorough introduction and analysis. In detail, we implemented the Graph Attention Network and several commonly used tricks to challenge the node classification task on the ogbn-arxiv dataset. Our experiments show satisfactory results on this dataset. At the same time, we explored the effectiveness of different tricks proposed or introduced in the paper. From our experiments, we found that Node2Vec embeddings, label usage, and C&amp;S achieved noticeable improvements on the task while changing the network structure of GAT and leveraging self-KD barely increases the accuracy.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/cs224w_21fall/";
            },},{id: "projects-chinese-dialogue-corpora-dstc9",
          title: 'Chinese Dialogue Corpora + DSTC9',
          description: "Three widely used open datasets for Chinese dialogue (CrossWOZ, KdConv, LCCC), each shipped with benchmark models and, for CrossWOZ, a user simulator. Together they have 3,000+ GitHub stars. I also co-organized the DSTC9 multi-domain task-completion track built on them.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/dialogue-datasets/";
            },},{id: "projects-llm-evaluation-amp-alignment",
          title: 'LLM Evaluation &amp;amp; Alignment',
          description: "Three threads from Microsoft AI. Rubric-based LLM-judge evaluation with human-in-the-loop metrics for Copilot&#39;s retrieval-augmented generation; post-training for multimodal retrieval models (Qwen-VL, CLIP-style); and an LLM teacher distilled into a lightweight E5 retriever that shipped after A/B testing.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/llm-evaluation/";
            },},{id: "projects-model-based-rl-for-dialogue-policy",
          title: 'Model-Based RL for Dialogue Policy',
          description: "Built SLBO and vanilla policy-gradient agents for a movie-ticket booking task, a chatbot environment and user-simulator sampling pipeline to train them, and a harness for comparing model-based deep RL algorithms on dialogue policy. Stanford UGVR with Tengyu Ma.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/rl-dialogue/";
            },},{id: "projects-transformer-notes",
          title: 'Transformer Notes',
          description: "A nine-page, step-by-step derivation of the Transformer that I wrote as a CS224N teaching assistant (Stanford, Winter 2022, with Christopher Manning). Queries, keys and values; scaled dot-product and multi-head attention; the encoder and decoder layers; positional encoding; masking; residual connections and layer norm; then the whole picture. Also mentored 10+ final-project teams and held weekly office hours.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/transformer-notes/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6B%61%69%6C%69@%63%73.%73%74%61%6E%66%6F%72%64.%65%64%75", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=OgXCn2UAAAAJ", "_blank");
        },
      },{
        id: 'social-semanticscholar',
        title: 'Semantic Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://www.semanticscholar.org/author/47942023", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/huangkaili", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/KailiHG", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/kailih", "_blank");
        },
      },{
        id: 'social-dblp',
        title: 'DBLP',
        section: 'Socials',
        handler: () => {
          window.open("https://dblp.org/pid/259/2408.html", "_blank");
        },
      },{
        id: 'social-orcid',
        title: 'ORCID',
        section: 'Socials',
        handler: () => {
          window.open("https://orcid.org/0009-0007-5393-3479", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
