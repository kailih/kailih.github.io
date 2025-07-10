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
  },{id: "nav-cv",
          title: "cv",
          description: "Welcome to Kaili Huang&#39;s CV. Click the PDF icon to view or download the full resume.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "Kaili&#39;s publications by categories in reversed chronological order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "Selected repositories that Kaili has led the initiatives or greatly contributed to.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-teaching",
          title: "teaching",
          description: "Kaili&#39;s experience as a teaching assistant (TA).",
          section: "Navigation",
          handler: () => {
            window.location.href = "/teaching/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A selection of interesting projects that Kaili worked on while studying at Stanford.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "news-we-are-excited-to-announce-that-our-latest-research-colbert-serve-efficient-multi-stage-memory-mapped-scoring-was-recently-accepted-and-presented-at-the-47th-european-conference-on-information-retrieval-ecir-2025-our-work-introduces-a-new-method-for-making-state-of-the-art-neural-search-more-efficient-and-scalable",
          title: 'We are excited to announce that our latest research, ColBERT-serve: Efficient Multi-Stage Memory-Mapped...',
          description: "",
          section: "News",},{id: "projects-engineering-effective-in-context-inputs-for-gpt-3-in-openqa",
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
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6B%61%69%6C%69.%68%75%61%6E%67@%73%74%61%6E%66%6F%72%64.%65%64%75", "_blank");
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
