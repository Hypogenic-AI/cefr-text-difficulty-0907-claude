# Literature Review: Modeling Text Difficulty Using CEFR Levels

## 1. Introduction / Background

The Common European Framework of Reference for Languages (CEFR) is the dominant international standard for describing language proficiency, organizing competence into six levels from A1 (beginner) through C2 (mastery). Originally developed by the Council of Europe to harmonize language education and assessment across member states, the CEFR has become the de facto standard for curriculum design, textbook grading, and language testing worldwide. Its six-level scale provides a shared vocabulary for educators, test developers, and learners, and it has been adopted in contexts well beyond Europe, including language programs in East Asia, the Middle East, and the Americas.

From a computational linguistics perspective, the CEFR presents a compelling operationalization of text difficulty. Unlike raw readability scores or grade-level equivalencies developed primarily for English-language education (e.g., Flesch-Kincaid, Dale-Chall), CEFR levels are designed to be language-independent and proficiency-oriented. They describe what a learner at a given level can understand, not merely surface properties of the text itself. This distinction is consequential: a text rated at B2 is one that a B2-level learner should be able to comprehend with reasonable effort, which implies that CEFR difficulty is fundamentally tied to the cognitive and linguistic demands that text places on a reader at a given proficiency stage. The framework thus provides a human-centered, pedagogically grounded target variable for computational models of text difficulty.

The central methodological question in this literature is whether text difficulty---as captured by CEFR levels---is best modeled through interpretable linguistic features or through black-box machine learning. The feature-based tradition draws on decades of readability research and second language acquisition (SLA) theory, constructing models from lexical frequency profiles, syntactic complexity measures, discourse coherence indicators, and psycholinguistic variables. These features are motivated by theories of how readers process text: frequent words are recognized faster, shorter dependency arcs reduce working memory load, and coherent discourse structure supports comprehension. The competing paradigm treats CEFR classification as a supervised learning problem amenable to deep neural networks, particularly pretrained language models such as BERT and its multilingual variants, which learn text representations end-to-end without explicit feature engineering.

This review surveys the empirical literature on both approaches, with particular attention to comparative studies that pit feature-based models against neural alternatives on the same data. The evidence, as will be shown, suggests that the performance gap between well-engineered linguistic features and state-of-the-art neural models is surprisingly modest, while traditional readability formulas and zero-shot large language model (LLM) prompting approaches fall substantially short of both.

## 2. Feature-Based Approaches to CEFR Classification

### 2.1 Lexical Features

Lexical features are consistently the most powerful single group of predictors for CEFR-level classification, a finding that recurs across languages, datasets, and modeling frameworks. The intuition is straightforward: lower-proficiency texts use more frequent, more familiar vocabulary, while higher-proficiency texts increasingly draw on rarer, more specialized, and more abstract words. This pattern is robust enough that lexical features alone often approach the performance of full-feature models.

Word frequency is the foundational lexical feature. Vajjala and Meurers (2012) demonstrated that lexical frequency, drawn from reference corpora, was the single strongest predictor of CEFR-graded text difficulty across a rich feature set of over 152 SLA-motivated variables. Pilán et al. (2016) confirmed this in Swedish, finding that lexical features alone achieved F=0.80 at the document level, compared to F=0.81 for a model using all available feature groups---a gap of just one percentage point. Type-token ratio (TTR) and its corrected variants capture lexical diversity, which tends to increase with proficiency level as learners command a broader range of vocabulary. Arnold et al. (2018) found that word token and word type counts were among the most important features in their gradient boosted tree models of CEFR difficulty.

CEFR-specific vocabulary resources have proven particularly valuable. The English Vocabulary Profile (EVP) assigns CEFR levels to individual words and phrases, enabling features such as the proportion of vocabulary at or above the target CEFR level. Xia et al. (2016) showed that EVP-derived features were especially important for modeling L2 text difficulty, since they capture not just frequency but pedagogical sequencing---the order in which vocabulary is typically taught to language learners. Similarly, the Kelly list provides frequency-ranked vocabulary for Swedish, and Pilán et al. (2016) used it to construct CEFR-aligned lexical features for their Swedish readability system.

### 2.2 Syntactic Complexity

Syntactic complexity captures the structural difficulty of sentences and is grounded in psycholinguistic theories of sentence processing. More complex syntax---longer dependencies, deeper embedding, more subordination---places greater demands on working memory and is accordingly associated with higher CEFR levels.

Parse tree depth measures how deeply nested a sentence's constituent structure is, with deeper trees reflecting more embedded clauses and more complex hierarchical organization. Dependency length, the linear distance between a head word and its dependent, captures a different dimension of syntactic difficulty: longer dependencies require readers to maintain information in working memory across more intervening material. Khallaf and Sharoff (2021) found that syntactic features ranked among the top 10 most important features for Arabic CEFR classification, with dependency-based measures contributing predictive power beyond what lexical features alone provided.

T-unit analysis (the minimal terminable unit) and measures of subordination (subordinate clause ratio, clauses per T-unit) have been used extensively in SLA research to characterize writing development across proficiency levels. Vajjala and Meurers (2012) drew on this tradition, incorporating a comprehensive set of syntactic complexity measures motivated by SLA theory. Pitler and Nenkova (2008), in their study of Wall Street Journal article readability, found that verb phrase (VP) production rules correlated at r=0.42 with human readability judgments, confirming that syntactic structure encodes meaningful difficulty information.

At the sentence level, syntactic features become more important relative to lexical features. Pilán et al. (2016) observed that the gap between lexical-only models and full-feature models widened from 1 percentage point at document level to nearly 7 percentage points at sentence level (56.8% vs. 63.4%), suggesting that syntactic and other features carry more marginal information when there is less text available to estimate lexical profiles.

### 2.3 Language Model Features

Statistical language model features operationalize the predictability of text, drawing on the psycholinguistic insight that processing difficulty is related to surprisal---the information-theoretic surprise associated with each word given its context. Text composed of high-probability sequences is easier to process; text with unexpected words and constructions is harder.

Xia et al. (2016) found that language model features were the single best-performing feature group for CEFR classification of Cambridge exam texts, achieving an accuracy of 0.714 as a standalone group. When combined with other features in a self-training semi-supervised framework, accuracy rose to 0.797. This result is notable because it demonstrates that the statistical structure of text---its n-gram predictability and perplexity---carries substantial information about CEFR difficulty, above and beyond what lexical frequency and syntactic complexity encode.

François and Fairon (2012) used statistical language models as a core component of their French-as-a-foreign-language readability system, combining n-gram probabilities with logistic regression to predict difficulty levels. Their work demonstrated the applicability of language model features beyond English, supporting the cross-linguistic generality of the approach.

Pitler and Nenkova (2008) reported that vocabulary-based language model features correlated at r=0.45 with human readability judgments, making them the second strongest individual feature group in their study, behind discourse relation features (r=0.48) but ahead of syntactic features (r=0.42). Their combined model achieved R²=0.776, demonstrating that language model features provide complementary information to syntactic and discourse measures.

Despite these promising results, language model surprisal remains under-explored specifically for CEFR classification. Most work using LM features has employed relatively simple n-gram models. The relationship between modern neural language model perplexity (from GPT-family or BERT-family models) and CEFR difficulty has received comparatively little direct investigation, representing a clear gap in the literature.

### 2.4 Discourse Features

Discourse-level features capture properties of text organization that extend beyond individual words and sentences, including coherence, entity continuity, and the use of discourse connectors. Pitler and Nenkova (2008) provided the most direct evidence for discourse features in readability modeling, finding that discourse relation features---operationalized through entity grids and discourse connectors---achieved the highest individual correlation with readability judgments (r=0.48) of any feature group tested. Their combined model with discourse, syntax, vocabulary, and language model features achieved R²=0.776, and discourse features contributed unique predictive variance.

Entity grids (Barzilay and Lapata, 2008), which track the grammatical roles of entities across sentences, provide a formal representation of text coherence. The distribution of entity transitions (e.g., subject-to-subject continuity vs. abrupt topic shifts) differs systematically between more and less readable texts, and these patterns align with CEFR levels to the extent that higher-level texts employ more complex discourse organization.

Discourse connectors and cohesive devices are pedagogically significant in CEFR-aligned instruction, where learners are expected to use progressively more sophisticated linking expressions. However, the computational modeling of these features for CEFR classification has been relatively limited compared to lexical and syntactic features, likely because they require more sophisticated NLP pipelines and discourse-parsed corpora that are not available for many languages.

### 2.5 Psycholinguistic Features

Psycholinguistic features capture properties of words that relate to how they are acquired and processed by learners, including concreteness, imageability, and age of acquisition (AoA). Concrete, imageable words with low ages of acquisition are learned earlier and processed more quickly than abstract, low-imageability words acquired later in development. These properties align with CEFR levels: A1-A2 texts tend to use concrete, everyday vocabulary, while C1-C2 texts increasingly employ abstract, specialized terms.

Vajjala and Meurers (2012) incorporated psycholinguistic features from databases such as the MRC Psycholinguistic Database, finding that they contributed to CEFR prediction alongside frequency and syntactic measures. These features are particularly valuable because they capture dimensions of word difficulty that raw frequency does not: a word may be infrequent in a general corpus but highly concrete and easily learned (e.g., "giraffe"), or frequent but abstract and difficult (e.g., "however").

### 2.6 Traditional Readability Formulas and Why They Fail for CEFR

Traditional readability formulas---Flesch Reading Ease, Flesch-Kincaid Grade Level, Dale-Chall, SMOG, and similar metrics---were developed in mid-20th-century American educational contexts to estimate the grade-level appropriateness of English texts. They rely on surface proxies such as average sentence length and average syllable count per word, which serve as rough indicators of syntactic and lexical complexity.

These formulas perform poorly for CEFR classification, a finding that is consistent across studies. Pilán et al. (2016) reported that LIX (a Swedish readability formula analogous to Flesch-Kincaid) was essentially useless for CEFR-level prediction. Vajjala and Meurers (2012) described traditional formulas as "crude proxies" for the linguistic dimensions they attempt to capture, arguing that CEFR levels require fine-grained feature modeling that surface formulas cannot provide. Pitler and Nenkova (2008) found that surface metrics---sentence length, word length---were among the weakest predictors of readability, substantially underperforming discourse, syntactic, and language model features.

The fundamental problem is that traditional formulas were designed for native-speaker reading in a single language and educational context. CEFR levels encode a multidimensional construct---proficiency-appropriate difficulty---that depends on vocabulary control, grammatical complexity, discourse structure, and topic familiarity in ways that sentence length and syllable count cannot capture. A text with short sentences and simple words may still be C1 if it requires sophisticated pragmatic inference; a text with long sentences may be B1 if the syntax is formulaic and the vocabulary is controlled. These distinctions are invisible to traditional formulas.

## 3. Neural and Embedding-Based Approaches

### 3.1 BERT Fine-Tuning for CEFR Classification

The advent of pretrained transformer language models, particularly BERT and its multilingual variants, has provided a powerful alternative to feature engineering for CEFR classification. These models learn contextual word representations from large corpora through self-supervised pretraining, and can be fine-tuned on downstream classification tasks with relatively modest amounts of labeled data.

Khallaf and Sharoff (2021) fine-tuned Arabic-BERT on a corpus of 22,740 sentences labeled with CEFR levels, achieving a macro-F1 of 0.80. This outperformed their SVM model with engineered features and XLM-R embeddings, which achieved F1=0.75, representing a 5-point advantage for the fine-tuned transformer. However, they also found that the feature-based model transferred better across domains, suggesting that neural models may overfit to surface distributional properties of their training data.

Imperial et al. (2025), in their large-scale UniversalCEFR benchmark, fine-tuned XLM-R on a 505,807-text multilingual corpus spanning 13 languages, achieving 62.8% macro-F1. This represented the best overall performance in their evaluation, but the margin over their best feature-based model (RF-ALLFEATS at 58.3%) was only 4.5 percentage points. More recent multilingual transformers such as ModernBERT and EuroBERT were not evaluated in UniversalCEFR but represent potential improvements given their expanded pretraining data and architectural refinements.

Lagutina et al. (2023) compared BERT fine-tuning to feature-based SVM classification on a Russian CEFR dataset, finding BERT at 69% accuracy versus SVC with stylometric features at 67%---a gap of just 2 percentage points. This vanishingly small margin raises questions about whether the additional computational cost and opacity of neural models are justified for CEFR classification when well-designed features perform nearly as well.

### 3.2 Graph Convolutional Networks for Joint Readability

Fujinuma and Hagiwara (2021) proposed a graph convolutional network (GCN) approach that jointly models word-level and document-level readability. Their key insight was that words and documents exist in a shared difficulty space: difficult words tend to appear in difficult documents, and this co-occurrence structure can be exploited by a graph-based model that propagates information between word and document nodes.

Their GCN achieved 79.6% document-level accuracy. Critically, an ablation study showed that BERT embeddings were essential to performance, with their removal causing an 11.2 percentage point drop. This finding suggests that the GCN's strength derives primarily from the rich contextual representations provided by pretrained language models rather than from the graph structure per se, lending further support to the importance of contextual language modeling for CEFR classification.

### 3.3 Prototype-Based Methods

Arase et al. (2022) introduced CEFR-SP, a prototype-based approach to sentence-level CEFR classification that learns representative examples (prototypes) for each CEFR level and classifies new sentences by similarity to these prototypes. Using a BERT-based architecture on a corpus of 17,676 sentences, CEFR-SP achieved a macro-F1 of 84.5%, dramatically outperforming a bag-of-words baseline (52.3%).

A particularly informative finding from Arase et al. was that sentence length, while a useful discriminator at lower proficiency levels (A1-A2 vs. B1), was insufficient to distinguish among higher levels (B1-B2-C1-C2). This result underscores the multidimensionality of CEFR difficulty: at higher levels, difficulty is driven by lexical sophistication, syntactic complexity, and discourse demands rather than sheer length, and models that rely on surface proxies will plateau.

### 3.4 LLM Prompting Approaches and Their Limitations

The emergence of large language models (LLMs) with strong zero-shot capabilities has prompted interest in using these models directly for CEFR classification through prompting. The appeal is obvious: if an LLM can assign CEFR levels without task-specific training data, it would drastically simplify the classification pipeline.

However, empirical results have been consistently disappointing. Imperial et al. (2025) evaluated Gemma3 prompting on their UniversalCEFR benchmark and obtained only 43.2% macro-F1---nearly 20 percentage points below the fine-tuned XLM-R (62.8%) and 15 percentage points below the best feature-based model (58.3%). This substantial underperformance suggests that CEFR-level classification requires more precise calibration than LLMs acquire through general pretraining. LLMs may have a broad understanding of text difficulty but lack the fine-grained ability to discriminate between adjacent CEFR levels (e.g., B1 vs. B2), where the differences are subtle and pedagogically specific.

## 4. Comparative Studies: Features vs. Neural

The most informative studies for our research hypothesis are those that directly compare feature-based and neural approaches on the same datasets and evaluation protocols. Taken together, they paint a consistent picture: neural models hold a modest advantage, but the gap is smaller than might be expected given the representational power of pretrained transformers.

### 4.1 UniversalCEFR (Imperial et al., 2025)

The most comprehensive comparison to date, UniversalCEFR evaluated models on 505,807 texts across 13 languages. The headline results were: XLM-R fine-tuning achieved 62.8% macro-F1, RF with all features achieved 58.3%, RF with top features achieved 57.9%, and Gemma3 prompting achieved 43.2%. The 4.5-point gap between the best neural and best feature-based models is statistically meaningful but practically modest, especially given that the feature-based model is interpretable and computationally lightweight. Notably, for Czech, the random forest feature-based model actually outperformed XLM-R, demonstrating that the neural advantage is not universal and may depend on language-specific factors such as training data representation in the pretrained model.

### 4.2 Arabic CEFR Classification (Khallaf & Sharoff, 2021)

In Arabic, fine-tuned Arabic-BERT achieved F1=0.80 compared to F1=0.75 for SVM with engineered features and XLM-R embeddings---a 5-point gap. However, the feature-based model demonstrated better cross-domain transfer, suggesting that linguistic features capture more generalizable properties of text difficulty while neural models may partially rely on domain-specific distributional cues.

### 4.3 EFCAMDAT Study (Arnold et al., 2018)

Arnold et al. (2018) used the large EFCAMDAT corpus of 41,000 texts to compare gradient boosted trees (GBT) with engineered features against LSTMs. The GBT model achieved 0.916 AUC for the A1-to-A2 classification task, and the LSTM did not outperform the feature-based approach. Word token and type counts were the most important features. This result is particularly significant because LSTMs, while less powerful than modern transformers, represent the class of neural sequence models, and their failure to improve on engineered features with a substantial training corpus supports the hypothesis that well-chosen features capture the essential dimensions of CEFR difficulty.

### 4.4 Russian CEFR Classification (Lagutina et al., 2023)

Lagutina et al. (2023) reported the smallest gap in the literature: SVC with stylometric features achieved 67% accuracy compared to 69% for BERT---a mere 2-point difference. This near-parity result, on a different language and with a different feature set, strengthens the case that the feature-neural gap for CEFR classification is genuinely small when features are well-designed.

### 4.5 Readability with Discourse Features (Pitler & Nenkova, 2008)

While not specifically targeting CEFR, Pitler and Nenkova (2008) provided foundational evidence for the importance of linguistically motivated features in readability modeling. Their combined model of discourse relations, syntax, vocabulary, and language model features achieved R²=0.776. Discourse features alone correlated at r=0.48 with readability, vocabulary LM features at r=0.45, and VP production rules at r=0.42. Surface metrics performed poorly. This study established that readability is best modeled through multiple linguistic dimensions, not surface proxies or single feature groups.

## 5. Cross-Lingual and Multilingual Findings

A critical question for CEFR classification is whether models and features generalize across languages. The CEFR framework is explicitly language-independent, but the linguistic features that predict CEFR levels may be language-specific in their effectiveness.

### 5.1 Universal CEFR Modeling (Vajjala & Rama, 2018)

Vajjala and Rama (2018) used the MERLIN corpus to investigate cross-lingual CEFR classification across German, Italian, and Czech. They found that word n-grams combined with domain features were the strongest within-language predictors (German 0.686, Italian 0.837), but that POS n-grams transferred best across languages, achieving 0.758 for Italian in a cross-lingual setting. Domain features alone were weak, suggesting that CEFR difficulty is primarily a function of linguistic complexity rather than topic. The superiority of POS n-grams in cross-lingual transfer is theoretically interesting because part-of-speech patterns abstract over language-specific vocabulary while capturing syntactic and morphological complexity patterns that recur across languages.

### 5.2 UniversalCEFR: 13 Languages (Imperial et al., 2025)

The UniversalCEFR benchmark dramatically expanded the cross-lingual evidence base to 13 languages. A key finding was that feature effectiveness is language-dependent: no single feature set dominates across all languages, and the relative importance of lexical, syntactic, and morphological features varies. For well-resourced languages with large pretraining corpora, XLM-R tends to outperform feature-based models; for lower-resource languages like Czech, feature-based models can be competitive or even superior. This pattern suggests that data quantity moderates the feature-neural gap, with features providing more robust performance when training data is limited.

### 5.3 ReadMe++ Multilingual Readability (Naous et al., 2023)

The ReadMe++ dataset aggregated readability-annotated texts from 112 sources across 5 languages, providing a large-scale multilingual benchmark. While not exclusively CEFR-focused, this resource highlights the growing recognition that readability research must move beyond English monolingual settings, and that multilingual evaluation is essential for assessing the generality of any proposed approach.

## 6. Key Findings and Research Gaps

### 6.1 Lexical Frequency Is the Most Powerful Single Feature

Across studies, languages, and datasets, lexical frequency emerges as the single most consistently powerful predictor of CEFR difficulty. Vajjala and Meurers (2012) identified it as the strongest predictor in their comprehensive SLA-motivated feature set. Pilán et al. (2016) showed that lexical features alone achieved 99% of the full model's document-level performance. Arnold et al. (2018) found word token and type counts---frequency-related measures---to be the most important features in gradient boosted trees. Xia et al. (2016) reported language model features (which are frequency-sensitive) as the best single feature group. This convergence across independent studies establishes lexical frequency as the bedrock of CEFR difficulty prediction.

### 6.2 The Features-vs.-Neural Gap Is Modest

The empirical gap between well-engineered feature-based models and fine-tuned neural models is consistently modest, typically in the range of 2 to 5 F1 or accuracy points. UniversalCEFR reports 4.5 points (58.3% vs. 62.8%); Khallaf and Sharoff report 5 points (0.75 vs. 0.80); Lagutina et al. report 2 points (67% vs. 69%); Arnold et al. report no neural advantage at all. This pattern suggests that feature-based models capture most of the information relevant to CEFR classification, and that neural models provide only incremental improvement---likely by capturing residual distributional patterns not explicitly encoded in the feature set.

### 6.3 Sentence-Level Classification Demands More Features

Document-level classification benefits from extensive text, which enables stable estimation of lexical profiles and other aggregate statistics. Sentence-level classification is substantially harder and more feature-hungry. Pilán et al. (2016) found that the gap between lexical-only and full-feature models widened from 1 point at document level to nearly 7 points at sentence level. Arase et al. (2022) showed that sentence length was insufficient to discriminate above B1 at the sentence level. These findings indicate that sentence-level CEFR classification requires richer feature representations that integrate syntactic, contextual, and potentially discourse-contextual information.

### 6.4 Data Quantity Moderates the Features-vs.-Neural Gap

The UniversalCEFR results suggest that the advantage of neural models is partially a function of data availability. For languages with abundant pretraining data and large fine-tuning corpora, neural models outperform features more convincingly. For lower-resource scenarios, feature-based models are competitive or superior (as seen with Czech in UniversalCEFR). This moderation effect has practical implications: in many real-world CEFR classification scenarios---particularly for less-studied languages or specialized domains---feature-based models may be the more reliable choice.

### 6.5 Traditional Readability Formulas Are Essentially Useless for CEFR

No study in this review found traditional readability formulas to be effective for CEFR classification. Pilán et al. (2016) found LIX useless. Vajjala and Meurers (2012) dismissed traditional formulas as crude proxies. Pitler and Nenkova (2008) found surface metrics to be the weakest predictors. The failure of these formulas reinforces the argument that CEFR difficulty is a multidimensional construct that cannot be reduced to sentence and word length.

### 6.6 LLM Zero-Shot Prompting Substantially Underperforms

Imperial et al. (2025) showed that Gemma3 prompting achieved only 43.2% macro-F1 on UniversalCEFR, nearly 20 points below fine-tuned XLM-R and 15 points below feature-based random forests. This result establishes that LLM prompting, at least with current models and prompting strategies, is not a viable approach to CEFR classification. The task requires calibrated discrimination between adjacent levels that general-purpose LLMs have not learned from pretraining alone.

### 6.7 Language Model Surprisal Is Under-Explored for CEFR

Despite the strong showing of language model features in Xia et al. (2016) and Pitler and Nenkova (2008), the systematic use of modern neural language model surprisal (e.g., GPT-based perplexity, BERT pseudo-log-likelihood) for CEFR classification remains under-explored. Most existing work uses classical n-gram language models. Given the strong theoretical motivation---surprisal captures processing difficulty, which is central to what CEFR levels encode---and the empirical success of LM features in readability research, this represents a significant gap and a promising direction for new work.

## 7. Implications for the Research Hypothesis

The hypothesis under investigation states that "linguistic factors such as lexical frequency, syntactic complexity, and language model surprisal contribute more to perceived text difficulty, as measured by CEFR levels, than treating text difficulty as a black-box machine learning label." The literature provides substantial, though nuanced, support for this hypothesis.

The strongest evidence in favor comes from the consistently modest gap between feature-based and neural models. If black-box models captured fundamentally different or richer information about CEFR difficulty, we would expect them to dramatically outperform feature-based models. Instead, the gap is 2-5 points across multiple independent studies, and in some cases (Arnold et al., 2018; Lagutina et al., 2023; Czech results in Imperial et al., 2025) there is no gap at all. This convergence suggests that the interpretable linguistic features---lexical frequency, syntactic complexity, and language model predictability---capture the great majority of the information relevant to CEFR classification.

Further support comes from the abysmal performance of approaches that ignore linguistic structure entirely. LLM prompting, which treats CEFR classification as a general reasoning task without task-specific features or training, achieves only 43.2% (Imperial et al., 2025). Traditional readability formulas, which use only the crudest surface proxies, are consistently found to be useless. These negative results reinforce that CEFR difficulty is grounded in specific linguistic dimensions---precisely the dimensions that feature-based models are designed to capture.

The evidence also supports the specific feature groups named in the hypothesis. Lexical frequency is the single strongest predictor across virtually all studies. Syntactic complexity contributes meaningfully, especially at the sentence level (Pilán et al., 2016) and in languages with rich morphosyntax (Khallaf & Sharoff, 2021). Language model features were the best single feature group in Xia et al. (2016) and the second strongest in Pitler and Nenkova (2008).

However, the hypothesis requires qualification in several respects. First, neural models do consistently outperform feature-based models, even if the margin is small. The 4.5-point advantage of XLM-R over RF-ALLFEATS in UniversalCEFR (Imperial et al., 2025) is modest but real, and it implies that pretrained representations capture some difficulty-relevant information that existing feature inventories miss. This residual information may correspond to distributional patterns too complex or too numerous to hand-engineer. Second, the advantage of feature-based models is most pronounced in low-resource settings; with abundant data and well-represented languages, neural models reliably edge ahead. Third, the hypothesis's claim about language model surprisal is currently supported more by analogy (from general readability research) than by direct evidence in CEFR-specific studies, given the under-exploration of neural surprisal features for this task.

In sum, the literature supports a strong version of the hypothesis for practical purposes: interpretable linguistic features, when carefully designed, explain most of the variance in CEFR difficulty and perform within a few points of state-of-the-art neural models. The weaker claim that features are strictly superior to black-box approaches is not supported---neural models do provide a small but consistent improvement. The most promising direction may be a hybrid approach that combines the interpretability and robustness of linguistic features with the representational richness of neural language models, a direction that several studies (Xia et al., 2016; Khallaf & Sharoff, 2021; Fujinuma & Hagiwara, 2021) have begun to explore.

## References

Arase, Y., Uchida, S., & Kajiwara, T. (2022). CEFR-Based Sentence Difficulty Annotation and Assessment Using Prototype Networks. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Association for Computational Linguistics.

Arnold, T., Ballier, N., Gaillat, T., & Lissón, P. (2018). Predicting CEFR levels in learner English on the basis of metrics and full texts. In *Proceedings of the BEA Workshop*. Association for Computational Linguistics.

François, T., & Fairon, C. (2012). An "AI readability" formula for French as a foreign language. In *Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL)*. Association for Computational Linguistics.

Fujinuma, Y., & Hagiwara, M. (2021). Joint Prediction of Word and Document Readability Using Graph Convolutional Networks. In *Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications (BEA)*. Association for Computational Linguistics.

Imperial, J. M., Tack, A., Francois, T., & Shardlow, M. (2025). UniversalCEFR: A Universal Benchmark for CEFR-Level Text Difficulty Classification Across Languages. *arXiv preprint*.

Khallaf, N., & Sharoff, S. (2021). Automatic Difficulty Classification of Arabic Sentences. In *Proceedings of the Sixth Arabic Natural Language Processing Workshop (WANLP)*. Association for Computational Linguistics.

Lagutina, N., Lagutina, K., Boychuk, E., & Nikiforova, N. (2023). Comparison of Machine Learning and BERT-Based Methods for CEFR Level Classification of Russian Texts. *Communications in Computer and Information Science*.

Naous, T., Ryan, M., & Xu, W. (2023). ReadMe++: Benchmarking Multilingual Language Models for Multi-Domain Readability Assessment. *arXiv preprint*.

Pilán, I., Volodina, E., & Zesch, T. (2016). Predicting proficiency levels in learner writings by transferring a linguistic complexity model from expert-written coursebooks. In *Proceedings of COLING 2016*. Association for Computational Linguistics.

Pitler, E., & Nenkova, A. (2008). Revisiting Readability: A Unified Framework for Predicting Text Quality. In *Proceedings of the 2008 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. Association for Computational Linguistics.

Vajjala, S., & Meurers, D. (2012). On Improving the Accuracy of Readability Classification Using Insights from Second Language Acquisition. In *Proceedings of the Seventh Workshop on Building Educational Applications Using NLP*. Association for Computational Linguistics.

Vajjala, S., & Rama, T. (2018). Experiments with Universal CEFR Classification. In *Proceedings of the Thirteenth Workshop on Innovative Use of NLP for Building Educational Applications (BEA)*. Association for Computational Linguistics.

Xia, M., Kochmar, E., & Briscoe, T. (2016). Text Readability Assessment for Second Language Learners. In *Proceedings of the 11th Workshop on Innovative Use of NLP for Building Educational Applications (BEA)*. Association for Computational Linguistics.
