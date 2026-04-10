# SciLint Score Calibration Set

20 papers (all ≤25 pages) with known ground truth for calibrating SciLint Score axes.
PDFs are not committed — download with URLs below.

## Domain Coverage

| Domain | Papers | Count |
|--------|--------|-------|
| CS / ML | Transformer, ResNet, BERT-finetune, GNN Survey | 4 |
| Physics | Graphene, LIGO, LK-99, Wilczek Time Crystals | 4 |
| Biology / Biochemistry | CRISPR, AlphaFold, Macchiarini, Baughman | 4 |
| Social Science / Psychology | LaCour, Camerer, Ritchie Null Results | 3 |
| Medicine / Clinical | RECOVERY RCT, Ioannidis, Shoukat | 3 |
| Earth / Environmental | Ceballos Sixth Extinction | 1 |
| Economics | Reinhart-Rogoff | 1 |

## Papers

### Positive controls (should score HIGH)

| # | File | Citation | Source URL | Ground truth |
|---|------|----------|-----------|-------------|
| 1 | novoselov2004_graphene.pdf | Novoselov & Geim (2004). "Electric Field Effect in Atomically Thin Carbon Films." Science 306(5696) | https://arxiv.org/pdf/cond-mat/0410550 | Nobel Prize 2010. High integrity, high contribution. |
| 2 | vaswani2017_attention.pdf | Vaswani et al. (2017). "Attention Is All You Need." NeurIPS 2017 | https://arxiv.org/pdf/1706.03762 | Foundational ML. High integrity, high contribution. |
| 3 | ligo2016_gravitational_waves.pdf | Abbott et al. (2016). "Observation of Gravitational Waves from a Binary Black Hole Merger." PRL 116(6) | https://arxiv.org/pdf/1602.03837 | Nobel Prize 2017. Very high across all axes. |
| 4 | he2015_resnet.pdf | He et al. (2015). "Deep Residual Learning for Image Recognition." CVPR 2016 | https://arxiv.org/pdf/1512.03385 | ~200K citations. True architectural innovation. |
| 5 | jinek2013_crispr_human.pdf | Jinek et al. (2013). "RNA-programmed genome editing in human cells." eLife 2:e00471 | https://cdn.elifesciences.org/articles/00471/elife-00471-v1.pdf | Nobel Prize 2020 (Doudna). CRISPR in human cells. Biology landmark. |
| 6 | jumper2021_alphafold.pdf | Jumper et al. (2021). "Highly accurate protein structure prediction with AlphaFold." Nature 596, 583-589 | https://link.springer.com/content/pdf/10.1038/s41586-021-03819-2.pdf | Nobel Prize 2024 (Hassabis). Solved protein folding. Chemistry/bio landmark. |

### Integrity failures — retracted / fabricated (should score LOW)

| # | File | Citation | Source URL | Ground truth |
|---|------|----------|-----------|-------------|
| 7 | shoukat2024_hallucinated_refs.pdf | Shoukat et al. (2024). "A comparative analysis of blended learning..." PLoS ONE 19(3). RETRACTED. | https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0298220&type=printable | 18/76 refs AI-hallucinated. Canonical AI-era citation fraud. |
| 8 | baughman2016_fabricated_retracted.pdf | Baughman et al. (2016). "...Inositol Pyrophosphate Kinase Inhibitors." PLoS ONE 11(10). RETRACTED. | https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0164378&type=printable | ORI-confirmed fabrication in 11 figures. Biology fraud. |
| 9 | lacour2014_fabricated_retracted.pdf | LaCour & Green (2014). "When contact changes minds." Science 346(6215). RETRACTED. | https://people.stat.sc.edu/Tebbs/stat110/lacour.pdf | Entire dataset fabricated. Statistical forensics detected. |
| 10 | macchiarini2014_fabricated_retracted.pdf | Jungebluth, Macchiarini et al. (2014). "...tissue-engineered oesophagus in rats." Nature Comms 5:3562. RETRACTED. | https://link.springer.com/content/pdf/10.1038/ncomms5562.pdf | Surgical fraud, no ethical permits, patients died. |
| 11 | toner2024_ai_productivity.pdf | Toner-Rodgers (2024). "AI, Scientific Discovery, and Product Innovation." arXiv:2412.17866v1. WITHDRAWN. | https://arxiv.org/pdf/2412.17866v1 | MIT found "no confidence in data." Recent AI-era fraud. |

### Methodological issues / bold claims without rigor (should score MIXED)

| # | File | Citation | Source URL | Ground truth |
|---|------|----------|-----------|-------------|
| 12 | lk99_2023.pdf | Kim et al. (2023). "The First Room-Temperature Ambient-Pressure Superconductor." | https://arxiv.org/pdf/2307.12008 | Extraordinary claims, no controls, not reproduced. |
| 13 | reinhart2010_debt.pdf | Reinhart & Rogoff (2010). "Growth in a Time of Debt." NBER w15639 | https://www.nber.org/system/files/working_papers/w15639/w15639.pdf | Famous Excel error. Text clean, data wrong. |
| 14 | lu2024_ai_scientist.pdf | Lu et al. (2024). "The AI Scientist." arXiv:2408.06292 | https://arxiv.org/pdf/2408.06292 | Real refs, overclaiming, cross-section inconsistency. |
| 15 | kaplan2020_scaling_laws.pdf | Kaplan et al. (2020). "Scaling Laws for Neural Language Models." arXiv:2001.08361 | https://arxiv.org/pdf/2001.08361 | Proprietary data but verifiable methodology. Novel predictions confirmed. |

### Replication / meta-science / null results (HIGH integrity, LOW Lakatos)

| # | File | Citation | Source URL | Ground truth |
|---|------|----------|-----------|-------------|
| 17 | camerer2018_replication.pdf | Camerer et al. (2018). "Evaluating the replicability of social science experiments." Nature Human Behaviour 2(9) | https://pure.eur.nl/ws/files/37359856/Camerer_et_al._2018_Evaluating_the_replicability_of_social_science_experiments_in_Nature_and_Science_between_2010_and_2015.pdf | 21 replications. High integrity, low progressiveness. |
| 18 | ioannidis2005_false_findings.pdf | Ioannidis (2005). "Why Most Published Research Findings Are False." PLoS Medicine 2(8) | https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.0020124&type=printable | Theoretical. High integrity, divergent contribution axes. |
| 19 | errington2021_replicability.pdf | Errington et al. (2021). "Challenges for assessing replicability in preclinical cancer biology." eLife 10:e67995 | https://elifesciences.org/articles/67995.pdf | Preregistered. 8-year cancer replication project. Only 50/193 experiments completable. |
| 20 | ritchie2012_failing_future.pdf | Ritchie et al. (2012). "Failing the Future: Three Unsuccessful Attempts to Replicate Bem's 'Retroactive Facilitation of Recall' Effect." PLoS ONE 7(3) | https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0033423&type=printable | Preregistered null result. High integrity, no positive finding. |

### Clinical trial (HIGH test severity)

| # | File | Citation | Source URL | Ground truth |
|---|------|----------|-----------|-------------|
| 21 | recovery2020_dexamethasone.pdf | RECOVERY Collaborative (2020). "Effect of Dexamethasone in Hospitalized Patients with COVID-19." medRxiv 2020.06.22.20137273 | https://www.medrxiv.org/content/10.1101/2020.06.22.20137273v1.full.pdf | Preregistered RCT, n=6425. First drug proven to reduce COVID mortality. |

### Review / survey (HIGH unification, ZERO experiments)

| # | File | Citation | Source URL | Ground truth |
|---|------|----------|-----------|-------------|
| 22 | wu2021_gnn_survey.pdf | Wu et al. (2021). "A Comprehensive Survey on Graph Neural Networks." IEEE TNNLS 32(1). arXiv:1901.00596 | https://arxiv.org/pdf/1901.00596 | ~4500 citations. No experiments. High unification, zero empirical content. |

### Incremental / unconventional / short

| # | File | Citation | Source URL | Ground truth |
|---|------|----------|-----------|-------------|
| 23 | sun2019_bert_finetune.pdf | Sun et al. (2019). "How to Fine-Tune BERT for Text Classification?" CCL 2019 | https://arxiv.org/pdf/1905.05583 | ~1500 citations. High integrity, low contribution. Incremental ML. |
| 24 | perelman2002_ricci_flow.pdf | Perelman (2002). "The entropy formula for the Ricci flow." arXiv:math/0211159 | https://arxiv.org/pdf/math/0211159 | Millennium Prize. Sparse refs, unconventional format. |
| 25 | wilczek2012_time_crystals.pdf | Wilczek (2012). "Quantum Time Crystals." PRL 109:160401. arXiv:1202.2539 | https://arxiv.org/pdf/1202.2539 | 4 pages. Nobel laureate. Bold theoretical proposal, no experiments. |

### Earth / environmental science

| # | File | Citation | Source URL | Ground truth |
|---|------|----------|-----------|-------------|
| 26 | ceballos2015_sixth_extinction.pdf | Ceballos et al. (2015). "Accelerated modern human-induced species losses." Science Advances 1(5) | https://www.echosciences-grenoble.fr/uploads/attachment/attached_file/23338500/6th_extinction-Ceballos-2015.pdf | Conservative methodology. Extinction rates 8-100x background. |

## Expected Rankings (ordinal constraints)

These constraints define "correct" calibration. The system passes when all hold.
**Fix the system, not the scores.** Never add logic targeting specific papers — only general improvements to how the scoring system reasons.

```
# Positive controls should beat all non-landmarks
LIGO > LK-99
ResNet > BERT-finetune
Graphene > Reinhart-Rogoff
AlphaFold > BERT-finetune
CRISPR > BERT-finetune

# Retracted papers should be bottom quartile
Shoukat < Camerer
Baughman < Camerer
LaCour < Ioannidis
Macchiarini < CRISPR

# Cross-domain: bio landmarks ≈ physics/CS landmarks
AlphaFold ≈ LIGO  (both should be top quartile)
CRISPR > LK-99

# True innovation > incremental
ResNet > BERT-finetune
Transformer > BERT-finetune

# Replication ≠ penalty
Camerer > BERT-finetune  (honest replication > honest incremental)

# Bold claims without rigor should not win
Graphene > LK-99
LIGO > LK-99

# Theoretical landmark should score well despite no experiments
Ioannidis > BERT-finetune

# Fraud should score below honest work
Shoukat < Ioannidis  (both meta-level, one has fake refs)

# Clinical RCT beats uncontrolled claims
RECOVERY > LK-99  (preregistered RCT > uncontrolled claims)
RECOVERY > Reinhart-Rogoff  (clinical trial > working paper with data error)

# Honest null results beat uncontrolled claims
Ritchie > LK-99  (rigorous null > extraordinary uncontrolled)
Ritchie > Shoukat  (honest null > fabricated positive)

# Survey has different axis profile
Wu-survey unification > Transformer unification
Wu-survey test-severity < any experimental paper
Wu-survey > Shoukat  (competent survey > fabricated paper)

# Short theoretical letter: high innovation, low severity
Wilczek progressiveness > BERT-finetune progressiveness
Wilczek test-severity < Graphene test-severity

# Earth science with conservative methodology
Ceballos > LK-99  (conservative methodology > extraordinary claims)
Ceballos > Shoukat  (honest analysis > fabricated refs)
```

## Download script

```bash
cd evals/calibration_data
# arXiv papers
curl -sL -o novoselov2004_graphene.pdf "https://arxiv.org/pdf/cond-mat/0410550"
curl -sL -o vaswani2017_attention.pdf "https://arxiv.org/pdf/1706.03762"
curl -sL -o ligo2016_gravitational_waves.pdf "https://arxiv.org/pdf/1602.03837"
curl -sL -o he2015_resnet.pdf "https://arxiv.org/pdf/1512.03385"
curl -sL -o lk99_2023.pdf "https://arxiv.org/pdf/2307.12008"
curl -sL -o lu2024_ai_scientist.pdf "https://arxiv.org/pdf/2408.06292"
curl -sL -o kaplan2020_scaling_laws.pdf "https://arxiv.org/pdf/2001.08361"
curl -sL -o sun2019_bert_finetune.pdf "https://arxiv.org/pdf/1905.05583"
curl -sL -o perelman2002_ricci_flow.pdf "https://arxiv.org/pdf/math/0211159"
curl -sL -o toner2024_ai_productivity.pdf "https://arxiv.org/pdf/2412.17866v1"
# NBER
curl -sL -o reinhart2010_debt.pdf "https://www.nber.org/system/files/working_papers/w15639/w15639.pdf"
# PLoS (open access)
curl -sL -o shoukat2024_hallucinated_refs.pdf "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0298220&type=printable"
curl -sL -o baughman2016_fabricated_retracted.pdf "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0164378&type=printable"
curl -sL -o ioannidis2005_false_findings.pdf "https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.0020124&type=printable"
# eLife (open access)
curl -sL -o jinek2013_crispr_human.pdf "https://cdn.elifesciences.org/articles/00471/elife-00471-v1.pdf"
# Springer/Nature (OA)
curl -sL -o jumper2021_alphafold.pdf "https://link.springer.com/content/pdf/10.1038/s41586-021-03819-2.pdf"
curl -sL -o macchiarini2014_fabricated_retracted.pdf "https://link.springer.com/content/pdf/10.1038/ncomms5562.pdf"
# Institutional repos
curl -sL -o camerer2018_replication.pdf "https://pure.eur.nl/ws/files/37359856/Camerer_et_al._2018_Evaluating_the_replicability_of_social_science_experiments_in_Nature_and_Science_between_2010_and_2015.pdf"
curl -sL -o lacour2014_fabricated_retracted.pdf "https://people.stat.sc.edu/Tebbs/stat110/lacour.pdf"
# eLife
curl -sL -o errington2021_replicability.pdf "https://elifesciences.org/articles/67995.pdf"
# PLoS ONE (null results)
curl -sL -o ritchie2012_failing_future.pdf "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0033423&type=printable"
# medRxiv (clinical trial)
curl -sL -o recovery2020_dexamethasone.pdf "https://www.medrxiv.org/content/10.1101/2020.06.22.20137273v1.full.pdf"
# arXiv (survey + short letter)
curl -sL -o wu2021_gnn_survey.pdf "https://arxiv.org/pdf/1901.00596"
curl -sL -o wilczek2012_time_crystals.pdf "https://arxiv.org/pdf/1202.2539"
# Science Advances mirror (earth science)
curl -sL -o ceballos2015_sixth_extinction.pdf "https://www.echosciences-grenoble.fr/uploads/attachment/attached_file/23338500/6th_extinction-Ceballos-2015.pdf"
```
