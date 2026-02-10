# Research Plan: Modeling Text Difficulty Using CEFR Levels

## Motivation & Novelty Assessment

### Why This Research Matters
Text difficulty assessment is central to language education, yet much recent NLP work treats CEFR classification as a pure prediction task—feeding text into BERT and accepting the label. This obscures *why* a text is difficult. Educators, curriculum designers, and learners benefit far more from knowing which linguistic factors drive difficulty than from a black-box label. Understanding the contribution of lexical frequency, syntactic complexity, and language model surprisal to CEFR levels can inform pedagogical material design and automated readability tools.

### Gap in Existing Work
The literature review reveals three key gaps:
1. **Modern neural surprisal is under-explored for CEFR**: Most feature-based work uses classical n-gram LMs. The relationship between GPT-2/BERT-based surprisal and CEFR levels has not been systematically studied at the sentence level.
2. **Feature importance in BERT is opaque**: While BERT achieves 84.5% macro-F1 on CEFR-SP (Arase et al., 2022), we don't know what linguistic properties it relies on. Probing/diagnostic experiments linking BERT predictions to interpretable features are missing.
3. **Direct comparison with diagnostic analysis**: Studies compare features vs. neural on accuracy, but rarely analyze *where and why* they disagree, or whether BERT's advantage comes from capturing specific linguistic dimensions beyond standard features.

### Our Novel Contribution
We conduct a systematic, multi-method analysis on the CEFR-SP English sentence dataset:
1. Extract comprehensive linguistic features (lexical, syntactic, surprisal) and train interpretable classifiers
2. Fine-tune a BERT-based classifier as the neural baseline
3. Perform diagnostic experiments: feature ablation, BERT probing via feature correlation, error analysis of model disagreements
4. Quantify how much of BERT's predictions can be explained by interpretable features (R² of feature regression on BERT logits)

### Experiment Justification
- **Experiment 1 (Feature extraction & EDA)**: Establishes which linguistic features correlate with CEFR levels and validates that our features capture meaningful difficulty dimensions.
- **Experiment 2 (Interpretable classifiers)**: Tests whether traditional features alone can approach neural performance, using logistic regression, random forest, and gradient boosted trees.
- **Experiment 3 (BERT fine-tuning)**: Provides the neural baseline to compare against.
- **Experiment 4 (Diagnostic experiments)**: The core novelty—probing BERT with feature regression, ablation studies, and error analysis to understand *what* each approach captures.

## Research Question
Can interpretable linguistic features (lexical frequency, syntactic complexity, language model surprisal) explain most of the variance in CEFR sentence difficulty, and where does a fine-tuned BERT classifier capture information beyond these features?

## Hypothesis Decomposition
- **H1**: Lexical frequency features are the single strongest predictor group for sentence-level CEFR classification.
- **H2**: Adding syntactic complexity and LM surprisal features significantly improves over lexical features alone.
- **H3**: A well-engineered feature-based model achieves within 5 F1 points of fine-tuned BERT.
- **H4**: Most of BERT's predictive behavior (>70% variance) can be explained by a linear combination of interpretable linguistic features.
- **H5**: BERT's advantage concentrates on distinguishing adjacent levels (e.g., B1 vs. B2) where feature-based models struggle most.

## Proposed Methodology

### Dataset
- **CEFR-SP** (Arase et al., 2022): 10,004 English sentences with CEFR labels (A1-C2)
- Distribution: A1=124, A2=1271, B1=3305, B2=3330, C1=1744, C2=230
- Note: Class imbalance—A1 and C2 are underrepresented. Will use stratified splits and macro-F1.

### Feature Groups

1. **Lexical features**: Word frequency (SubtlexUS), type-token ratio, word length stats, word-level CEFR stats (via cefrpy), difficult word ratio
2. **Syntactic features**: Dependency tree depth, dependency distance, POS distribution, subordinate clause ratio, sentence length
3. **LM Surprisal features**: GPT-2 mean/max surprisal per sentence, GPT-2 perplexity
4. **Traditional readability**: Flesch-Kincaid, ARI, Coleman-Liau (as weak baselines)

### Approach
1. **Feature extraction**: Use LFTK, TextDescriptives, surprisal library, cefrpy, textstat
2. **Interpretable models**: Logistic Regression (L2), Random Forest, XGBoost with feature groups ablation
3. **BERT baseline**: Fine-tune `bert-base-uncased` on CEFR-SP with classification head
4. **Diagnostic experiments**:
   a. Feature ablation: Train models with each feature group removed
   b. BERT probing: Regress BERT's predicted class probabilities on interpretable features
   c. Error analysis: Compare where feature model and BERT disagree
   d. Confusion matrix analysis: Which level pairs are hardest for each approach

### Baselines
- Majority class baseline
- Sentence length only baseline
- Traditional readability formulas only
- Each individual feature group

### Evaluation Metrics
- **Macro F1** (primary): Handles class imbalance, standard in CEFR literature
- **Accuracy**: For comparability with prior work
- **Per-class F1**: To identify level-specific strengths/weaknesses
- **Adjacent accuracy**: Fraction of predictions within 1 CEFR level (captures near-misses)

### Statistical Analysis Plan
- 5-fold stratified cross-validation for all models
- Report mean ± std for all metrics
- McNemar's test for pairwise model comparison
- Bonferroni correction for multiple comparisons
- Feature importance via permutation importance (model-agnostic)
- Spearman correlation between feature values and ordinal CEFR level

## Expected Outcomes
- Lexical features will be the strongest single group (~55-60% macro-F1 alone)
- Surprisal features will add 3-5 points over lexical alone
- Full feature model: ~65-72% macro-F1
- BERT: ~75-80% macro-F1
- Feature regression on BERT logits: R² > 0.70
- Largest errors will be on adjacent levels (B1↔B2) for both approaches

## Timeline and Milestones
1. Environment setup & data loading (10 min)
2. Feature extraction pipeline (30 min)
3. EDA & feature analysis (20 min)
4. Interpretable model training & ablation (30 min)
5. BERT fine-tuning (30 min)
6. Diagnostic experiments (30 min)
7. Visualization & analysis (20 min)
8. Report writing (30 min)

## Potential Challenges
- **Class imbalance**: A1 (124) and C2 (230) are very small. Mitigation: macro-F1, stratified CV, possibly class weights.
- **Feature extraction speed**: LFTK/spaCy on 10K sentences may be slow. Mitigation: batch processing.
- **BERT training time**: Mitigation: leverage GPU (RTX A6000 available), use small learning rate and early stopping.
- **Surprisal computation**: GPT-2 inference on 10K sentences. Mitigation: batch with GPU.

## Success Criteria
1. All feature groups extracted and validated
2. At least 3 interpretable classifiers trained and evaluated
3. BERT fine-tuned and evaluated on same splits
4. Feature ablation completed showing relative importance
5. BERT probing analysis completed
6. Clear answer to whether features explain most of BERT's predictions
