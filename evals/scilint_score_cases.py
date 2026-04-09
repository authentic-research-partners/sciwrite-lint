"""Ground-truth cases for SciLint Score taxonomy and Laudan evaluation.

Separated from scilint_score_eval.py to keep that file under 1000 lines.
Each generator returns a list of dataclass instances with known labels.
"""

from __future__ import annotations

from evals.scilint_score_types import LaudanCase, TaxonomyCase


# ---------------------------------------------------------------------------
# Taxonomy cases: claims with known ground truth
# ---------------------------------------------------------------------------


def generate_taxonomy_cases() -> list[TaxonomyCase]:
    """Generate claim taxonomy evaluation cases.

    Coverage target: at least 3 cases per enum value across all 5 dimensions.
    Cases use naturalistic context (no curated METHODS SUMMARY headers).
    Ground truth reflects expert consensus. No overlap with prompt examples.
    """
    return [
        # =================================================================
        # PREDICTIONS (novel results from this paper)
        # =================================================================
        TaxonomyCase(
            name="pred_quant_severe",
            claim_text=(
                "Our pruning method achieves 92.3% of the dense model's "
                "accuracy while reducing parameters by 85%."
            ),
            key="han2023",
            context=(
                "We evaluate structured pruning on ResNet-50, EfficientNet-B4, "
                "and DeiT-S. For each model, we compare against magnitude "
                "pruning, lottery ticket, and SynFlow baselines at the same "
                "sparsity level. We run each configuration with 3 seeds and "
                "report mean accuracy. We also ablate the pruning schedule "
                "(linear, cosine, cubic) and the scoring criterion (L1, L2, "
                "gradient). Our pruning method achieves 92.3% of the dense "
                "model's accuracy while reducing parameters by 85%."
            ),
            expected={
                "type": "prediction",
                "specificity": "quantified",
                "testability": "falsifiable",
                "support": "severe_test",
                "scope": "within_domain",
            },
        ),
        TaxonomyCase(
            name="pred_quant_weak",
            claim_text=(
                "The fine-tuned model reaches 78.6 F1 on the biomedical NER benchmark."
            ),
            key="lee2023",
            context=(
                "We fine-tune a pretrained language model on the BC5CDR "
                "dataset using standard hyperparameters from the original "
                "paper. The fine-tuned model reaches 78.6 F1 on the "
                "biomedical NER benchmark. We did not tune hyperparameters "
                "or compare against other recent methods."
            ),
            expected={
                "type": "prediction",
                "specificity": "quantified",
                "testability": "falsifiable",
                "support": "weak_test",
                "scope": "within_domain",
            },
        ),
        TaxonomyCase(
            name="pred_dir_severe",
            claim_text=(
                "Graph neural networks significantly outperform "
                "feature-engineered baselines on molecular property prediction."
            ),
            key="gilmer2020",
            context=(
                "We compare five GNN variants (GCN, GAT, GIN, SchNet, DimeNet) "
                "against random forest and XGBoost with RDKit fingerprints "
                "across 12 MoleculeNet datasets. Statistical significance is "
                "assessed via paired t-tests with Bonferroni correction over "
                "10 random splits. We ablate message-passing depth, pooling "
                "strategy, and feature representation. Graph neural networks "
                "significantly outperform feature-engineered baselines on "
                "molecular property prediction."
            ),
            expected={
                "type": "prediction",
                "specificity": "directional",
                "testability": "falsifiable",
                "support": "severe_test",
                "scope": "within_domain",
            },
        ),
        TaxonomyCase(
            name="pred_quant_notest",
            claim_text=(
                "We conjecture that training on 10x more multilingual data "
                "would close the 12-point gap with English performance."
            ),
            key="conneau2022",
            context=(
                "Our multilingual model underperforms the English-only "
                "variant by 12 points on average. We conjecture that "
                "training on 10x more multilingual data would close the "
                "12-point gap with English performance. Verifying this "
                "would require resources beyond our current compute budget."
            ),
            expected={
                "type": "prediction",
                "specificity": "quantified",
                "testability": "falsifiable",
                "support": "no_test",
                "scope": "within_domain",
            },
        ),
        TaxonomyCase(
            name="pred_dir_weak",
            claim_text=(
                "Curriculum learning improves convergence speed compared "
                "to random data ordering."
            ),
            key="bengio2009",
            context=(
                "We train the same architecture with and without curriculum "
                "ordering on CIFAR-10. Curriculum learning improves "
                "convergence speed compared to random data ordering. "
                "A single training run was performed for each condition."
            ),
            expected={
                "type": "prediction",
                "specificity": "directional",
                "testability": "falsifiable",
                "support": "weak_test",
                "scope": "within_domain",
            },
        ),
        # =================================================================
        # EXPLANATIONS (why something happens)
        # =================================================================
        TaxonomyCase(
            name="expl_posthoc_paper",
            claim_text=(
                "We attribute the unexpected accuracy drop on the validation "
                "set to distribution shift caused by temporal data leakage."
            ),
            key="koh2021",
            context=(
                "Unexpectedly, validation accuracy was 8 points lower than "
                "test accuracy. After investigating, we attribute the "
                "unexpected accuracy drop on the validation set to "
                "distribution shift caused by temporal data leakage. No "
                "controlled experiment was run to confirm this hypothesis."
            ),
            expected={
                "type": "explanation",
                "specificity": "directional",
                "testability": "falsifiable",
                "support": "post_hoc",
                "scope": "within_paper",
            },
        ),
        TaxonomyCase(
            name="expl_vague_unfals",
            claim_text=(
                "Attention mechanisms have become an indispensable component "
                "of modern sequence modeling."
            ),
            key="vaswani2017",
            context=(
                "Since the introduction of the Transformer, attention "
                "mechanisms have become an indispensable component of "
                "modern sequence modeling, replacing recurrence in most "
                "state-of-the-art systems."
            ),
            expected={
                "type": "explanation",
                "specificity": "vague",
                "testability": "unfalsifiable",
                "support": "no_test",
                "scope": "within_domain",
            },
        ),
        TaxonomyCase(
            name="expl_dir_notest",
            claim_text=(
                "The performance improvement likely arises from the model's "
                "ability to capture higher-order feature interactions."
            ),
            key="cheng2023",
            context=(
                "Our model outperforms the shallow baseline. The performance "
                "improvement likely arises from the model's ability to "
                "capture higher-order feature interactions, though we did "
                "not verify this with a controlled ablation."
            ),
            expected={
                "type": "explanation",
                "specificity": "directional",
                "testability": "falsifiable",
                "support": "post_hoc",
                "scope": "within_paper",
            },
        ),
        # =================================================================
        # REPRODUCTIONS (replicating previously published results)
        # =================================================================
        TaxonomyCase(
            name="reprod_quant_severe",
            claim_text=(
                "We verify the reported 45.1 mAP of YOLOv8 on COCO and "
                "measure 44.8 mAP under identical conditions."
            ),
            key="jocher2023",
            context=(
                "To validate the original claims, we download the official "
                "YOLOv8 weights and evaluate on COCO val2017 with the same "
                "preprocessing pipeline. We verify the reported 45.1 mAP "
                "and measure 44.8 mAP under identical conditions. We repeat "
                "evaluation 5 times with different hardware to rule out "
                "non-determinism. The 0.3-point gap is within the expected "
                "variance reported by the original authors."
            ),
            expected={
                "type": "reproduction",
                "specificity": "quantified",
                "testability": "falsifiable",
                "support": "severe_test",
                "scope": "within_domain",
            },
        ),
        TaxonomyCase(
            name="reprod_dir_weak",
            claim_text=(
                "We confirm that data augmentation improves robustness "
                "as reported by Hendrycks et al."
            ),
            key="hendrycks2019",
            context=(
                "Following the setup of Hendrycks et al., we confirm that "
                "data augmentation improves robustness. We tested a single "
                "augmentation strategy on ImageNet-C without varying the "
                "corruption types or severity levels."
            ),
            expected={
                "type": "reproduction",
                "specificity": "directional",
                "testability": "falsifiable",
                "support": "weak_test",
                "scope": "within_domain",
            },
        ),
        TaxonomyCase(
            name="reprod_quant_notest",
            claim_text=(
                "We plan to replicate the 3.2% WER reported by Radford "
                "et al. on LibriSpeech test-clean."
            ),
            key="radford2023",
            context=(
                "Radford et al. report 3.2% WER on LibriSpeech test-clean "
                "with Whisper-large. We plan to replicate this result as "
                "part of our upcoming benchmark suite but have not yet "
                "conducted the evaluation."
            ),
            expected={
                "type": "reproduction",
                "specificity": "quantified",
                "testability": "falsifiable",
                "support": "no_test",
                "scope": "within_domain",
            },
        ),
        # =================================================================
        # SYNTHESIS (combining ideas into new frameworks)
        # =================================================================
        TaxonomyCase(
            name="synth_cross_severe",
            claim_text=(
                "Our compiler-aware neural architecture search integrates "
                "hardware cost models from compiler optimization with "
                "differentiable NAS from deep learning."
            ),
            key="wu2023",
            context=(
                "We bring together hardware-aware compilation (from the "
                "systems community) and differentiable NAS (from deep "
                "learning). We evaluate on 4 hardware targets (GPU, TPU, "
                "mobile CPU, edge NPU) and compare against both NAS-only "
                "and compiler-only baselines. We ablate the compiler cost "
                "model component and the NAS search space independently."
            ),
            expected={
                "type": "synthesis",
                "specificity": "directional",
                "testability": "falsifiable",
                "support": "severe_test",
                "scope": "cross_domain",
            },
        ),
        TaxonomyCase(
            name="synth_cross_notest",
            claim_text=(
                "We propose a theoretical framework connecting information "
                "bottleneck theory to the generalization properties of "
                "overparameterized networks."
            ),
            key="shwartz2022",
            context=(
                "Our framework links information-theoretic compression "
                "(from information theory) to implicit regularization "
                "(from learning theory). This is a purely theoretical "
                "contribution; empirical validation is left to future work."
            ),
            expected={
                "type": "synthesis",
                "specificity": "directional",
                "testability": "falsifiable",
                "support": "no_test",
                "scope": "cross_domain",
            },
        ),
        TaxonomyCase(
            name="synth_domain_weak",
            claim_text=(
                "Our method combines contrastive learning with masked "
                "language modeling into a single pretraining objective."
            ),
            key="chi2023",
            context=(
                "We unify contrastive and generative pretraining by "
                "jointly optimizing both losses. We compare against "
                "contrastive-only and MLM-only baselines on GLUE. "
                "A single training run was used for all experiments."
            ),
            expected={
                "type": "synthesis",
                "specificity": "directional",
                "testability": "falsifiable",
                "support": "weak_test",
                "scope": "within_domain",
            },
        ),
        # =================================================================
        # TAUTOLOGICAL / UNFALSIFIABLE
        # =================================================================
        TaxonomyCase(
            name="taut_definition",
            claim_text=(
                "An unsupervised method, by definition, does not use "
                "labeled examples during training."
            ),
            key="goodfellow2016",
            context=(
                "We clarify the terminology used throughout this survey. "
                "An unsupervised method, by definition, does not use "
                "labeled examples during training, though it may use "
                "structural information from the data distribution."
            ),
            expected={
                "type": "explanation",
                "specificity": "vague",
                "testability": "tautological",
                "support": "no_test",
                "scope": "within_domain",
            },
        ),
        TaxonomyCase(
            name="taut_logical",
            claim_text=(
                "A model that memorizes the training set will necessarily "
                "achieve zero training loss."
            ),
            key="zhang2021",
            context=(
                "We note that a model that memorizes the training set will "
                "necessarily achieve zero training loss, but this tells us "
                "nothing about its ability to generalize."
            ),
            expected={
                "type": "explanation",
                "specificity": "vague",
                "testability": "tautological",
                "support": "no_test",
                "scope": "within_domain",
            },
        ),
        TaxonomyCase(
            name="unfals_scope",
            claim_text=(
                "Artificial intelligence is poised to reshape every "
                "industry in the coming decades."
            ),
            key="mckinsey2024",
            context=(
                "Given the pace of progress, artificial intelligence is "
                "poised to reshape every industry in the coming decades. "
                "The exact timeline and nature of this transformation "
                "remain uncertain."
            ),
            expected={
                "type": "prediction",
                "specificity": "vague",
                "testability": "unfalsifiable",
                "support": "no_test",
                "scope": "cross_domain",
            },
        ),
        TaxonomyCase(
            name="unfals_hedge",
            claim_text=("More data will always help, given sufficient model capacity."),
            key="hestness2017",
            context=(
                "Scaling laws suggest that more data will always help, given "
                "sufficient model capacity. However, practical constraints "
                "on data quality and compute may limit returns."
            ),
            expected={
                "type": "explanation",
                "specificity": "directional",
                "testability": "unfalsifiable",
                "support": "no_test",
                "scope": "within_domain",
            },
        ),
        # =================================================================
        # WITHIN_PAPER scope
        # =================================================================
        TaxonomyCase(
            name="paper_specific_quant",
            claim_text=(
                "In our setup, batch size 4096 yields 1.2% higher accuracy "
                "than batch size 256."
            ),
            key="goyal2017",
            context=(
                "We sweep batch sizes from 256 to 8192 on our specific "
                "pipeline. In our setup, batch size 4096 yields 1.2% "
                "higher accuracy than batch size 256. This was observed "
                "only with our learning rate schedule and may not transfer "
                "to other configurations. We tested 3 batch sizes total."
            ),
            expected={
                "type": "prediction",
                "specificity": "quantified",
                "testability": "falsifiable",
                "support": "weak_test",
                "scope": "within_paper",
            },
        ),
        TaxonomyCase(
            name="paper_specific_posthoc",
            claim_text=(
                "We speculate that the training instability we observed "
                "at epoch 50 was caused by the learning rate warmup "
                "interacting with our custom data loader."
            ),
            key="chen2023",
            context=(
                "Training diverged at epoch 50 in 2 of 5 runs. We speculate "
                "that the training instability we observed was caused by "
                "the learning rate warmup interacting with our custom data "
                "loader. We did not investigate this further as it did not "
                "affect final accuracy."
            ),
            expected={
                "type": "explanation",
                "specificity": "vague",
                "testability": "falsifiable",
                "support": "post_hoc",
                "scope": "within_paper",
            },
        ),
        # =================================================================
        # ADVERSARIAL / EDGE CASES
        # =================================================================
        TaxonomyCase(
            name="negative_result",
            claim_text=(
                "Contrary to prior work, we find that knowledge distillation "
                "does not improve performance on low-resource languages."
            ),
            key="wu2024",
            context=(
                "We apply knowledge distillation from mBERT to smaller "
                "models for 15 low-resource languages. Contrary to prior "
                "work, we find that knowledge distillation does not improve "
                "performance. We compare 4 distillation methods (soft "
                "targets, intermediate layers, attention transfer, CRD) "
                "against direct fine-tuning on 3 seeds each."
            ),
            expected={
                "type": "prediction",
                "specificity": "directional",
                "testability": "falsifiable",
                "support": "severe_test",
                "scope": "within_domain",
            },
        ),
        TaxonomyCase(
            name="hedged_claim",
            claim_text=(
                "Our results suggest that sparse attention may be sufficient "
                "for most practical NLP tasks."
            ),
            key="child2019",
            context=(
                "Across 6 NLP benchmarks, sparse attention patterns achieve "
                "within 0.5% of full attention. Our results suggest that "
                "sparse attention may be sufficient for most practical NLP "
                "tasks. We note this requires further validation on "
                "generation tasks."
            ),
            expected={
                "type": "prediction",
                "specificity": "directional",
                "testability": "falsifiable",
                "support": "weak_test",
                "scope": "within_domain",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Laudan cases: intro + limitations with expected score ranges
# ---------------------------------------------------------------------------


def generate_laudan_cases() -> list[LaudanCase]:
    """Generate Laudan problem-solving evaluation cases.

    Each case tests whether the LLM can correctly assess the ratio of
    problems solved to unacknowledged limitations.
    """
    return [
        LaudanCase(
            name="laudan_strong_honest",
            description="3 problems solved, 3 limitations honestly acknowledged",
            intro_text=(
                "Current neural machine translation systems suffer from "
                "three key limitations: (1) poor performance on low-resource "
                "languages, (2) inability to handle code-switching, and "
                "(3) high computational cost during inference. We address "
                "all three problems through a unified multilingual "
                "architecture with efficient decoding."
            ),
            limitations_text=(
                "Our approach has several limitations. First, while we "
                "improve low-resource translation, languages with fewer "
                "than 1,000 parallel sentences remain challenging. Second, "
                "our efficient decoding trades 2% quality for 3x speed — "
                "this may not be acceptable for high-stakes applications. "
                "Third, we only evaluate on Indo-European languages; "
                "generalization to typologically distant languages is "
                "untested."
            ),
            expected_min=0.4,
            expected_max=1.0,
        ),
        LaudanCase(
            name="laudan_no_limitations",
            description="Claims to solve problems but has no limitations section",
            intro_text=(
                "We present a novel framework that solves the long-standing "
                "problem of catastrophic forgetting in continual learning. "
                "Our method also addresses the related challenge of "
                "forward transfer between tasks."
            ),
            limitations_text="",
            expected_min=0.0,
            expected_max=0.5,
        ),
        LaudanCase(
            name="laudan_balanced",
            description="1 problem solved, acknowledges reasonable limitations",
            intro_text=(
                "Existing text summarization systems produce factually "
                "inconsistent summaries 30% of the time. We propose a "
                "verification module that reduces factual errors by 60%."
            ),
            limitations_text=(
                "Our verification module adds 40% latency to the "
                "summarization pipeline. Additionally, our approach "
                "relies on a separately trained NLI model which itself "
                "has approximately 85% accuracy, meaning some factual "
                "errors may still pass undetected."
            ),
            expected_min=0.3,
            expected_max=0.9,
        ),
        LaudanCase(
            name="laudan_overclaiming",
            description="Grand claims, minimal limitations acknowledgment",
            intro_text=(
                "We solve artificial general intelligence by training a "
                "single model on all available internet data. Our system "
                "achieves human-level performance on every benchmark "
                "and surpasses expert-level reasoning in all domains."
            ),
            limitations_text=(
                "We note that our model requires significant computational "
                "resources to train."
            ),
            expected_min=0.0,
            expected_max=0.4,
        ),
        LaudanCase(
            name="laudan_minor_contribution",
            description="Small incremental improvement, honest about scope",
            intro_text=(
                "We propose a simple modification to the Adam optimizer "
                "that improves convergence on noisy gradients. The change "
                "involves adding a momentum correction term."
            ),
            limitations_text=(
                "Our modification only helps when gradient noise is high "
                "(SNR < 1). On standard benchmarks with clean gradients, "
                "the improvement is negligible. We have only tested on "
                "image classification; applicability to NLP and RL is "
                "unknown. Our analysis assumes i.i.d. gradient noise, "
                "which may not hold in practice."
            ),
            expected_min=0.2,
            expected_max=0.8,
        ),
        LaudanCase(
            name="laudan_theoretical_only",
            description="Pure theory paper, no experiments",
            intro_text=(
                "We prove that neural networks with ReLU activations "
                "can approximate any Lipschitz function to arbitrary "
                "precision with O(1/epsilon^d) parameters."
            ),
            limitations_text=(
                "Our bounds are worst-case and may be loose for "
                "practical networks. The construction is not "
                "computationally efficient. We do not address "
                "how to find these networks via gradient descent."
            ),
            expected_min=0.2,
            expected_max=0.8,
        ),
        LaudanCase(
            name="laudan_many_problems_thin_limits",
            description="Claims many solved problems, thin limitations",
            intro_text=(
                "We address five critical challenges in federated learning: "
                "(1) communication efficiency, (2) non-IID data, "
                "(3) client drift, (4) privacy guarantees, and "
                "(5) Byzantine fault tolerance."
            ),
            limitations_text=(
                "Our experiments use synthetic non-IID partitions which "
                "may not reflect real-world heterogeneity."
            ),
            expected_min=0.0,
            expected_max=0.6,
        ),
        LaudanCase(
            name="laudan_survey_paper",
            description="Survey/position paper, no claimed solutions",
            intro_text=(
                "This survey reviews 150 papers on graph neural networks "
                "published between 2018 and 2024. We organize the "
                "literature along three axes: architecture, training "
                "methodology, and application domain."
            ),
            limitations_text=(
                "Our survey is limited to English-language publications "
                "from major venues. We may have missed relevant preprints "
                "and workshop papers. Our taxonomy is one of many possible "
                "organizational schemes."
            ),
            expected_min=0.0,
            expected_max=0.5,
        ),
    ]
