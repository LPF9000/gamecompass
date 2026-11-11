# Recommendation research

Research checked 2026-11-10. Algorithm expectations below are hypotheses, not measured results for this application. See the generated benchmark report for actual runs.

## Framework and compatibility

[RecBole](https://recbole.io/) provides a shared data representation, evaluation, and implementations across general, sequential, context-aware, and knowledge-aware recommendation. RecBole 2.0 describes related extension toolkits rather than a requirement to deploy a separate service. Its [source license](https://github.com/RUCAIBox/RecBole/blob/master/LICENSE) is MIT.

The inspected [1.2.1 setup](https://github.com/RUCAIBox/RecBole/blob/master/setup.py) requires PyTorch >=1.10 and Ray <=2.6.3. These old dependency constraints need an isolated research environment; a modern Python interpreter alone does not prove compatibility. Test the actual locked environment. Do not silently label a custom implementation as a RecBole experiment. Keep training optional so the local inference application can run without the research stack.

PyTorch supports CPU, CUDA and MPS, but support for a backend does not establish that every sparse operation used by a recommender supports it. LightGCN sparse graph propagation needs a real backend test. EASE and sparse item-neighbor baselines can use NumPy/SciPy on CPU. [PyTorch backend documentation](https://pytorch.org/docs/stable/notes/mps.html).

## Candidate algorithms

Let U = users, I = items, E = observed positive interactions, d = latent dimension, L = propagation layers, K = retained neighbors. Complexity is an engineering approximation for common implementations, not a measured runtime.

| Approach | Assumptions and requirements | Training / serving | Strengths | Limitations and cold start |
|---|---|---|---|---|
| Popularity | Count training positives only | O(E) / O(I) ranking | Essential zero-history comparator | No personalization; favors exposure and head items |
| ItemKNN | Co-preference reflects future preferences; user-item positives | Sparse Gram matrix; worst dense O(I²) storage, top-K O(IK) / sparse history aggregation | Fast immediate fold-in, interpretable evidence | Co-occurrence bias; no representation for unseen items |
| EASE | Regularized linear reconstruction of sparse positives | Gram O(E × average history), inversion O(I³), O(I²) memory / history × I | Strong simple baseline; no new-user retraining | Dense item matrix can exceed RAM; no cold-item signal |
| BPR-MF | Observed positives outrank sampled unobserved items | O(epochs × sampled triples × d) / O(Id) | Compact factors, directly optimizes pairwise rank | Unobserved is not dislike; sampled objective introduces assumptions; user fold-in required |
| LightGCN | Preference graph neighborhoods convey information | Roughly O(epochs × LEd), plus BPR optimization / O(Id) after embedding export | Collaborative graph structure without unnecessary nonlinear layers | More compute; graph leakage if held-out users/edges participate; new-user fold-in not automatic |
| Structured/TF-IDF content | Descriptions and tags carry preference signal | Sparse feature extraction / vector similarity | New items can be scored locally; transparent overlaps | Text similarity is not enjoyment; missing and promotional metadata |
| Sentence Transformer | Pretrained semantic space useful for game meaning | Batched transformer inference / vector dot products | Semantic wording, cold-item coverage | Generic language model may miss gameplay nuance; download and storage required |
| Feature ranker / hybrid | Validation users represent deployment; candidate features available at prediction time | Linear/logistic or tree ranking on validation-only examples / candidates × features | Can combine explicit dislikes, content and collaboration | Overfit, exposure bias and calibration risk; arbitrary blends are not evidence |
| Sequential SASRec | Order and recent actions predict later interaction | Self-attention generally O(sequence length²), trainable item embeddings | Tests short-term vs persistent preference | Review time is not play sequence; sparse users; new-user encoder must consume revealed prefix only |
| Knowledge-aware | Reliable entity relations improve preference signal | Depends on graph/model; graph construction material cost | Franchise, developer, gameplay connections | Missing or noisy knowledge graph; entity resolution can dominate quality |

[BPR paper](https://arxiv.org/abs/1205.2618) defines a pairwise ranking objective. Its negative sampling should be described as an optimization convention, not a negative preference label.

[LightGCN paper](https://arxiv.org/abs/2002.02126) motivates simplified linear graph propagation. Its published gains do not imply gains on this dataset or for previously unseen local users.

[EASE paper](https://arxiv.org/abs/1905.03375) establishes a closed-form regularized linear baseline and demonstrates that deep complexity is not automatically beneficial.

[SASRec paper](https://arxiv.org/abs/1808.09781) provides the sequential comparison. Do not interpret a sequence of review dates as a complete sequence of games played.

[all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) identifies Apache-2.0 weights and 384-dimensional embeddings. It is an optional local content experiment; store the exact model revision and do not make an unverified claim that it improves ranking.

## New-user engineering

An application user is never assumed to be one of the population dataset users. ItemKNN/EASE fold a sparse signed preference vector into item scores immediately. MF can freeze item factors and solve a regularized least-squares user vector using positives and explicit negatives, or optimize a small user vector. These are different objectives from the original BPR loss and must be benchmarked accordingly. Content vectors can aggregate positive and negative item features. Keep ownership and playtime separate and lower-confidence than explicit ratings. A newly described game can enter content candidates with high uncertainty even if it has no collaborative factor.

## Evaluation decisions

Use fixed deterministic splits and full eligible-catalog ranking, excluding revealed interactions. Unobserved candidates are unknown, not proved negatives. Separate whole-user holdouts from warm-user splits; remove cold users from population training and item statistics. Fix catalog availability using training-time information for temporal experiments. Do not use later tag/description/review totals in a purported strict temporal study without a snapshot.

Select hyperparameters and hybrid weights on validation users only, lock them, then report test users once. Bootstrap users for confidence intervals and use paired user differences when comparing models. Report the number of eligible users at every revealed-history length; never pad short histories to claim a 100-history evaluation. Include niche-item recall, coverage, novelty and diversity alongside Recall/NDCG/MAP. Offline positive-review retrieval is not calibrated enjoyment prediction, nor proof of satisfaction from unplayed recommendations.

The immediate priorities are trustworthy fold-in, explicit negative feedback, metadata quality, user-disjoint evaluation and comparative baselines. Sequential and knowledge-aware models should be promoted only after they beat these controls under matched conditions.
