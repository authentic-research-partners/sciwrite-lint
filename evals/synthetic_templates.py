"""Realistic LaTeX paper templates for synthetic evaluation.

Provides a complete, self-consistent CS conference paper about a fictional
method "AdaptiveAttend" for text classification. Each section can be
individually overridden to inject specific errors for evaluation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Section content blocks (clean, self-consistent)
# ---------------------------------------------------------------------------

CLEAN_ABSTRACT = (
    "We present AdaptiveAttend, a novel attention mechanism for text "
    "classification that dynamically adjusts attention weights based on "
    "input complexity. Our method achieves 92.3\\% accuracy on SST-2, "
    "representing a 5.4\\% relative improvement over the previous "
    "state-of-the-art. On the IMDB benchmark, AdaptiveAttend obtains "
    "94.1\\% accuracy while using 30\\% fewer parameters than comparable "
    "transformer models. We further demonstrate that our approach "
    "generalizes to multilingual settings, achieving competitive results "
    "on the XNLI benchmark across 15 languages without language-specific "
    "fine-tuning. Our analysis reveals that the adaptive mechanism learns "
    "to allocate more attention heads to syntactically complex sentences, "
    "providing interpretable evidence for its improved performance."
)

CLEAN_INTRO = (
    "Text classification remains a fundamental challenge in natural language "
    "processing, with applications ranging from sentiment analysis "
    "\\cite{pang2008} to document categorization \\cite{joachims1998} and "
    "spam detection \\cite{sahami1998}. Recent advances in transformer-based "
    "models \\cite{vaswani2017} have dramatically improved performance across "
    "a wide range of NLP benchmarks, establishing new state-of-the-art results "
    "on tasks such as sentiment analysis, natural language inference, and "
    "question answering \\cite{devlin2019}.\n\n"
    "However, current attention mechanisms treat all inputs uniformly, "
    "allocating the same computational budget regardless of input complexity. "
    "This is suboptimal: a simple sentence like ``The movie was great'' requires "
    "far less processing than a complex, multi-clause sentence with negation and "
    "sarcasm. Prior work on adaptive computation \\cite{graves2016} has explored "
    "variable-depth processing, but these approaches operate at the layer level "
    "rather than the attention level.\n\n"
    "In this paper, we make three key contributions:\n"
    "\\begin{enumerate}\n"
    "\\item We propose AdaptiveAttend, an attention mechanism that dynamically "
    "adjusts the number of active attention heads based on a learned complexity "
    "score (Section~\\ref{sec:methods}).\n"
    "\\item We introduce a complexity-aware training objective that encourages "
    "the model to use minimal computation for simple inputs while preserving "
    "full capacity for difficult ones (Section~\\ref{sec:training}).\n"
    "\\item We present extensive experiments on three benchmarks showing that "
    "AdaptiveAttend achieves state-of-the-art accuracy with 30\\% fewer "
    "parameters on average (Section~\\ref{sec:experiments}).\n"
    "\\end{enumerate}"
)

CLEAN_RELATED = (
    "\\paragraph{Transformer architectures.} "
    "The transformer architecture \\cite{vaswani2017} introduced self-attention "
    "as a replacement for recurrence, enabling parallel computation across "
    "sequence positions. BERT \\cite{devlin2019} demonstrated that pre-training "
    "bidirectional transformers on large corpora produces representations that "
    "transfer effectively to downstream tasks. Subsequent work explored more "
    "efficient attention patterns, including sparse attention \\cite{child2019} "
    "and linear attention \\cite{katharopoulos2020}.\n\n"
    "\\paragraph{Adaptive computation.} "
    "Graves~\\cite{graves2016} introduced adaptive computation time (ACT), "
    "allowing recurrent networks to vary the number of processing steps per "
    "input. Universal Transformers \\cite{dehghani2019} applied this idea to "
    "the transformer architecture, using a halting mechanism to determine "
    "per-position depth. Our work differs by adapting the \\emph{breadth} "
    "(number of attention heads) rather than depth.\n\n"
    "\\paragraph{Efficient transformers.} "
    "DistilBERT \\cite{sanh2019} demonstrated that knowledge distillation can "
    "compress BERT to 60\\% of its size with minimal quality loss. MobileBERT "
    "\\cite{sun2020} further optimized for mobile deployment. Unlike distillation "
    "approaches, AdaptiveAttend achieves efficiency through input-dependent "
    "computation rather than static compression."
)

CLEAN_METHODS_ARCH = (
    "AdaptiveAttend modifies the standard multi-head attention mechanism by "
    "introducing a lightweight gating network that determines which attention "
    "heads to activate for each input. Given an input sequence "
    "$\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$, we first compute a "
    "complexity score:\n"
    "\\begin{equation}\\label{eq:complexity}\n"
    "c = \\sigma(\\mathbf{w}^\\top \\text{pool}(\\mathbf{X}) + b)\n"
    "\\end{equation}\n"
    "where $\\text{pool}(\\cdot)$ denotes mean pooling over the sequence "
    "dimension, $\\mathbf{w} \\in \\mathbb{R}^d$ and $b \\in \\mathbb{R}$ "
    "are learned parameters, and $\\sigma$ is the sigmoid function. The "
    "complexity score $c \\in [0, 1]$ controls the number of active heads:\n"
    "\\begin{equation}\\label{eq:heads}\n"
    "k = \\lceil c \\cdot H \\rceil\n"
    "\\end{equation}\n"
    "where $H$ is the total number of attention heads. Only the top-$k$ "
    "heads (ranked by their learned importance scores) are activated. "
    "The architecture is shown in Figure~\\ref{fig:architecture}.\n\n"
    "\\begin{figure}[t]\n"
    "\\centering\n"
    "\\fbox{\\parbox{0.8\\linewidth}{\\centering "
    "[AdaptiveAttend Architecture Diagram]}}\n"
    "\\caption{Overview of AdaptiveAttend. The gating network computes a "
    "complexity score that determines how many attention heads are active "
    "for each input.}\n"
    "\\label{fig:architecture}\n"
    "\\end{figure}"
)

CLEAN_METHODS_TRAINING = (
    "We train AdaptiveAttend with a composite objective:\n"
    "\\begin{equation}\\label{eq:loss}\n"
    "\\mathcal{L} = \\mathcal{L}_{\\text{task}} + "
    "\\lambda \\mathcal{L}_{\\text{efficiency}}\n"
    "\\end{equation}\n"
    "where $\\mathcal{L}_{\\text{task}}$ is the standard cross-entropy loss "
    "and $\\mathcal{L}_{\\text{efficiency}} = \\frac{1}{N} \\sum_{i=1}^N c_i$ "
    "penalizes high complexity scores. The hyperparameter "
    "$\\lambda$ controls the efficiency--accuracy trade-off. We use "
    "$\\lambda = 0.01$ in all experiments (see Table~\\ref{tab:hyperparams}).\n\n"
    "\\begin{table}[t]\n"
    "\\centering\n"
    "\\caption{Hyperparameters for AdaptiveAttend training.}\n"
    "\\label{tab:hyperparams}\n"
    "\\begin{tabular}{ll}\n"
    "\\toprule\n"
    "Parameter & Value \\\\\n"
    "\\midrule\n"
    "Learning rate & $1 \\times 10^{-4}$ \\\\\n"
    "Batch size & 32 \\\\\n"
    "Epochs & 50 \\\\\n"
    "Attention heads ($H$) & 8 \\\\\n"
    "Hidden dimension ($d$) & 512 \\\\\n"
    "Efficiency weight ($\\lambda$) & 0.01 \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}"
)

CLEAN_DATASETS = (
    "We evaluate on three benchmarks:\n"
    "\\begin{itemize}\n"
    "\\item \\textbf{SST-2} \\cite{socher2013}: Binary sentiment classification "
    "with 67,349 training and 872 test examples.\n"
    "\\item \\textbf{IMDB} \\cite{maas2011}: Binary sentiment with 25,000 "
    "training and 25,000 test examples.\n"
    "\\item \\textbf{XNLI} \\cite{conneau2018}: Natural language inference "
    "across 15 languages, with 392,702 training examples.\n"
    "\\end{itemize}\n"
    "We compare against BERT-base \\cite{devlin2019}, DistilBERT "
    "\\cite{sanh2019}, and the Universal Transformer \\cite{dehghani2019}."
)

CLEAN_RESULTS = (
    "Table~\\ref{tab:results} summarizes our main results. AdaptiveAttend "
    "achieves 92.3\\% accuracy on SST-2, surpassing BERT-base (87.6\\%) by "
    "a margin of 4.7 percentage points, a 5.4\\% relative improvement. "
    "On IMDB, our model reaches 94.1\\%, compared to 91.3\\% for BERT-base. "
    "Importantly, AdaptiveAttend uses an average of 5.6 out of 8 attention "
    "heads per input, achieving 30\\% parameter reduction during inference.\n\n"
    "\\begin{table}[t]\n"
    "\\centering\n"
    "\\caption{Test accuracy (\\%) on classification benchmarks. "
    "Best results in \\textbf{bold}.}\n"
    "\\label{tab:results}\n"
    "\\begin{tabular}{lccc}\n"
    "\\toprule\n"
    "Model & SST-2 & IMDB & XNLI (avg) \\\\\n"
    "\\midrule\n"
    "BERT-base & 87.6 & 91.3 & 73.2 \\\\\n"
    "DistilBERT & 85.2 & 89.4 & 70.8 \\\\\n"
    "Universal Transformer & 88.1 & 91.7 & 74.0 \\\\\n"
    "\\midrule\n"
    "AdaptiveAttend (ours) & \\textbf{92.3} & \\textbf{94.1} & "
    "\\textbf{76.5} \\\\\n"
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table}\n\n"
    "On XNLI, AdaptiveAttend achieves 76.5\\% average accuracy across "
    "15 languages, outperforming BERT-base by 3.3 points. We observe "
    "particularly strong gains on morphologically complex languages such as "
    "Turkish and Finnish, where the adaptive mechanism activates more heads "
    "to handle agglutinative word forms. See Figure~\\ref{fig:heads} for "
    "the distribution of active heads across languages.\n\n"
    "\\begin{figure}[t]\n"
    "\\centering\n"
    "\\fbox{\\parbox{0.8\\linewidth}{\\centering "
    "[Head Activation Distribution by Language]}}\n"
    "\\caption{Average number of active attention heads per language on "
    "XNLI. Morphologically complex languages activate more heads.}\n"
    "\\label{fig:heads}\n"
    "\\end{figure}"
)

CLEAN_CONCLUSION = (
    "We presented AdaptiveAttend, an attention mechanism that dynamically "
    "adjusts the number of active attention heads based on input complexity. "
    "Our approach achieves 92.3\\% accuracy on SST-2, a 5.4\\% relative "
    "improvement over BERT-base, while using 30\\% fewer parameters on "
    "average. Experiments on IMDB and XNLI confirm that these gains "
    "generalize across domains and languages.\n\n"
    "Our analysis reveals that the adaptive mechanism learns linguistically "
    "meaningful patterns: syntactically complex sentences with negation, "
    "subordinate clauses, and long-range dependencies consistently activate "
    "more attention heads. This provides interpretable evidence that the "
    "model allocates computation where it is most needed.\n\n"
    "Future work includes extending AdaptiveAttend to generation tasks "
    "such as machine translation and summarization, where input complexity "
    "varies significantly across source sentences."
)

CLEAN_BIBITEMS = [
    r"\bibitem{vaswani2017} Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. In \emph{NeurIPS}.",
    r"\bibitem{devlin2019} Devlin, J., Chang, M., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers. In \emph{NAACL-HLT}.",
    r"\bibitem{pang2008} Pang, B. and Lee, L. (2008). Opinion Mining and Sentiment Analysis. \emph{Foundations and Trends in Information Retrieval}, 2(1--2):1--135.",
    r"\bibitem{joachims1998} Joachims, T. (1998). Text Categorization with Support Vector Machines. In \emph{ECML}.",
    r"\bibitem{sahami1998} Sahami, M., Dumais, S., Heckerman, D., and Horvitz, E. (1998). A Bayesian Approach to Filtering Junk E-Mail. In \emph{AAAI Workshop on Learning for Text Categorization}.",
    r"\bibitem{graves2016} Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. \emph{arXiv:1603.08983}.",
    r"\bibitem{child2019} Child, R., Gray, S., Radford, A., and Sutskever, I. (2019). Generating Long Sequences with Sparse Transformers. \emph{arXiv:1904.10509}.",
    r"\bibitem{katharopoulos2020} Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. (2020). Transformers Are RNNs. In \emph{ICML}.",
    r"\bibitem{dehghani2019} Dehghani, M., Gouws, S., Vinyals, O., et al. (2019). Universal Transformers. In \emph{ICLR}.",
    r"\bibitem{sanh2019} Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). DistilBERT, a Distilled Version of BERT. \emph{arXiv:1910.01108}.",
    r"\bibitem{sun2020} Sun, Z., Yu, H., Song, X., et al. (2020). MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices. In \emph{ACL}.",
    r"\bibitem{socher2013} Socher, R., Perelygin, A., Wu, J., et al. (2013). Recursive Deep Models for Semantic Compositionality. In \emph{EMNLP}.",
    r"\bibitem{maas2011} Maas, A., Daly, R., Pham, P., et al. (2011). Learning Word Vectors for Sentiment Analysis. In \emph{ACL}.",
    r"\bibitem{conneau2018} Conneau, A., Rinott, R., Lample, G., et al. (2018). XNLI: Evaluating Cross-lingual Sentence Representations. In \emph{EMNLP}.",
]


# ---------------------------------------------------------------------------
# Paper template
# ---------------------------------------------------------------------------

_PAPER_TEMPLATE = r"""\documentclass{{article}}
\usepackage{{amsmath,amssymb,graphicx,hyperref,booktabs}}

\title{{AdaptiveAttend: Dynamic Attention Head Allocation for Text Classification}}
\author{{Alice Chen \and Bob Nakamura \and Carol Petrov}}
\date{{}}

\begin{{document}}
\maketitle

\begin{{abstract}}
{abstract}
\end{{abstract}}

\section{{Introduction}}\label{{sec:intro}}
{intro}

\section{{Related Work}}\label{{sec:related}}
{related}

\section{{Proposed Approach}}\label{{sec:methods}}

\subsection{{Architecture}}\label{{sec:architecture}}
{methods_arch}

\subsection{{Training Procedure}}\label{{sec:training}}
{methods_training}

\section{{Experiments}}\label{{sec:experiments}}

\subsection{{Datasets and Baselines}}
{datasets}

\subsection{{Results}}
{results}

\section{{Conclusion}}\label{{sec:conclusion}}
{conclusion}

\begin{{thebibliography}}{{99}}
{bibliography}
\end{{thebibliography}}
\end{{document}}
"""


def build_realistic_paper(
    abstract: str = CLEAN_ABSTRACT,
    intro: str = CLEAN_INTRO,
    related: str = CLEAN_RELATED,
    methods_arch: str = CLEAN_METHODS_ARCH,
    methods_training: str = CLEAN_METHODS_TRAINING,
    datasets: str = CLEAN_DATASETS,
    results: str = CLEAN_RESULTS,
    conclusion: str = CLEAN_CONCLUSION,
    bibitems: list[str] | None = None,
) -> str:
    """Build a realistic LaTeX paper, overriding specific sections."""
    bib = bibitems if bibitems is not None else CLEAN_BIBITEMS
    return _PAPER_TEMPLATE.format(
        abstract=abstract,
        intro=intro,
        related=related,
        methods_arch=methods_arch,
        methods_training=methods_training,
        datasets=datasets,
        results=results,
        conclusion=conclusion,
        bibliography="\n".join(bib),
    )
