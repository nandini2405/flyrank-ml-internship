# Capstone Report —

- **Author:** Nandini Kancharla
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/nandini2405/flyrank-ml-internship
- **Date:** September 27, 2026

## 0. Abstract


This capstone asks which content pages should be reviewed first for refresh, expansion, protection, pruning, or monitoring based on observable search-performance signals. The analysis uses the FlyRank internship warehouse release, with daily content-performance data from January through June 2026 and page-level search signals including impressions, clicks, CTR, average position, and position bucket. A transparent CTR-versus-position benchmark baseline is compared with Logistic Regression and Decision Tree ranking models, using earlier monthly pairs for training and May → June 2026 as a held-out future-period test set. On the held-out period, Logistic Regression achieved **Precision@20 of 0.80, Precision@50 of 0.74, and Average Precision of 0.647**, compared with **0.30, 0.26, and 0.332** for the baseline and **0.70, 0.80, and 0.605** for the Decision Tree. The resulting Logistic Regression ranking is used as a **human review-prioritization tool**, helping reviewers identify pages to inspect earlier while leaving the final content action to human judgment and making no causal claims about future search performance.


## 1. Problem framing


This capstone supports the decision of **which pages should be reviewed first for refresh, expansion, protection, pruning, monitoring, or no action** based on observable search-performance signals.

The **unit of analysis is a content page**. The output is a **review-priority score and ranking** that helps identify pages that may warrant earlier human review.

A reviewer can use the ranking to prioritize limited review time, inspect the page and its search-performance context, and then decide whether any action is appropriate. The model does not automatically determine which action should be taken.

A wrong prioritization has two main costs. Reviewing a page that does not require attention can consume limited editorial resources, while failing to prioritize a page that does require attention can delay its review. The goal is therefore to improve how review effort is prioritized rather than to automate content decisions.

Data and machine learning are useful because the analysis considers multiple observable search-performance signals across many pages. A ranking model can combine these signals into a consistent review-priority score, allowing reviewers to focus attention on higher-priority pages first.

The analysis is intended as **decision support**. A high model score indicates higher priority for review under the evaluation design; it does not establish that a page has a specific underlying problem, that a particular content action is required, or that taking such an action will improve future search performance.

## 2. Data safety


This analysis uses the **FlyRank internship warehouse release** available through the gated Hugging Face dataset.

### Data used

The main source table is:

* `fact_content_daily_performance`

The table is at a **daily × client × content** grain. The analysis aggregates these daily records by client and content page.

The primary observable search-performance signals used are:

* Search impressions
* Search clicks
* Search position

These are used to calculate:

* Total impressions
* Total clicks
* CTR
* Average search position

The analysis uses **March 2026** for the current-window baseline, covering `2026-03-01` through `2026-03-31`. The March extract contains **9,841,378 daily rows**.

For the temporal analysis, consecutive monthly partitions from **January through June 2026** are used:

* January → February
* February → March
* March → April
* April → May
* May → June

The May → June pair is used as the future-period test set.

### Data excluded

The analysis deliberately excludes:

* Client names or domains
* URLs
* Private search queries
* Credentials or private information
* Raw exports
* Product flags
* Future-window performance metrics
* The starter `trend_direction` field as a baseline input

Pseudonymous identifiers such as `client_hash_id` and `content_hash_id` are used for grouping and matching pages across monthly partitions, rather than as predictive features.

### Leakage controls

Future-month performance information is used only to construct the target for the temporal supervised analysis. It is **not included as a model feature**.

The temporal target is whether next-month CTR is lower than current-month CTR. Therefore, next-month CTR and other next-month performance measures are kept out of the feature set.

The notebook also checks for potentially problematic columns including:

* `trend_direction`
* `future_ctr`
* `future_impressions`
* `future_clicks`
* `product_flag`

The purpose of these checks is to prevent information about the future outcome or production decisions from entering the ranking features.

The final supervised feature set contains only current-month signals:

* `impressions`
* `clicks`
* `current_ctr`
* `avg_position`
* `position_bucket`

### Public-safety framing

The analysis is designed as **public-safe, pseudonymized, decision-support analysis**. It does not use client names, domains, URLs, private search queries, credentials, or raw exports, and the final reviewer-facing queue uses the pseudonymous `content_hash_id` rather than publicly identifying a client or page.

These restrictions are intentional: the output is meant to demonstrate a reproducible review-prioritization analysis without exposing client-identifying or private information.

## 3. Baseline


The baseline provides a simple, interpretable way to identify pages that may represent a CTR review opportunity without using a machine-learning model.

### Baseline rule

A page is included in the baseline candidate set when:

1. It has at least **500 current-month impressions**.
2. Its CTR is below the CTR benchmark for its position bucket.

The analysis divides pages into four broad position buckets:

* Top 3
* 4–10
* 11–20
* 20+

For each position bucket, the benchmark CTR is calculated as:

> **Total clicks in the position bucket ÷ total impressions in the position bucket**

This creates a position-specific reference point rather than comparing all pages against a single overall CTR.

### CTR gap

For each page, the CTR opportunity gap is calculated as:

`position-benchmark CTR − page CTR`

A positive gap means that the page's observed CTR is below the benchmark for its position bucket.

Only pages with a positive gap are included in the baseline ranking.

### Baseline ranking score

The baseline score combines the CTR gap with the page's search visibility:

`log1p(impressions) × CTR gap`

The logarithmic transformation of impressions allows visibility to contribute to the ranking while reducing the influence of extremely large impression counts.

The resulting score is used to rank the candidate pages from highest to lowest review priority.

### Reviewer output

The baseline produces a transparent Top-20 review queue containing:

* Rank
* Pseudonymous content identifier
* Impressions
* Clicks
* CTR
* Average position
* Position bucket
* Position-benchmark CTR
* CTR gap
* Baseline score
* Reason code
* Suggested review action

The baseline reason code is:

`LOW_CTR_VS_POSITION_BENCHMARK`

and the suggested review action is:

`REVIEW_CTR_OPPORTUNITY`

These are review prompts rather than automatic content decisions.

### Why this is the baseline

The rule is intentionally simple and directly interpretable. A reviewer can understand why a page entered the queue by comparing its CTR with the benchmark for its position context and considering its search visibility.

The supervised models are evaluated against this baseline on the same held-out evaluation period to determine whether combining the available signals provides a different ranking of pages with subsequent CTR declines.

## 4. Model / analysis


### Feature set

The supervised models use information available during the current month:

* `impressions`
* `clicks`
* `current_ctr`
* `avg_position`
* `position_bucket`

The position bucket is treated as a categorical feature, while the other four variables are numerical.

Future-month performance information is not included in the model features.

### Target definition

The supervised models predict:

> **Whether a page's CTR declines in the following month.**

This target is stored as `target_ctr_declined`.

A value of `1` indicates that the page's CTR declined in the following month, while `0` indicates that it did not decline.

The target is constructed using future-month information only for labeling the training and test examples; those future performance values are not provided to the models as input features.

### Temporal modelling design

The data is divided into consecutive monthly pairs.

The training data uses:

* January → February 2026
* February → March 2026
* March → April 2026
* April → May 2026

The **May → June 2026** pair is kept as the held-out future-period test set.

This design allows the models to be trained on earlier observations and evaluated on a later period.

### Logistic Regression

The primary supervised model is **Logistic Regression**.

The numerical features are standardized using `StandardScaler`, while `position_bucket` is converted into categorical indicator variables using `OneHotEncoder`.

These transformations and the classifier are combined in a single scikit-learn pipeline.

The Logistic Regression model uses `max_iter=1000`.

For each held-out page, the model produces a probability for the positive class, representing the model's estimated probability of the target CTR-decline outcome. This probability is stored as the page's `score` and used to rank pages for review.

### Decision Tree

A **Decision Tree Classifier** is also evaluated as a second supervised approach.

The same preprocessing pipeline and feature set are used. The tree is constrained to:

`max_depth=3`

and uses `random_state=42`.

The predicted probability of the positive class is used as the Decision Tree ranking score.

### Model interpretation

The Logistic Regression coefficients are used to calculate feature contributions to each page's model score.

For each held-out page, the transformed feature values are multiplied by the learned coefficients to estimate their contribution to the model's linear predictor (log-odds).

The strongest positive contributions are converted into reviewer-facing reason codes.

These reason codes explain which observed model signals were most associated with a page receiving a higher score. They are **model associations, not causal explanations**.

### Output

The final analysis produces a ranked reviewer queue using the Logistic Regression score.

The queue contains the page's pseudonymous identifier, model score, current search-performance signals, position context, and model-derived reason codes.

The ranking is intended to help a human reviewer decide **which pages to inspect first**. It does not automatically determine whether a page should be refreshed, expanded, protected, pruned, monitored, or left unchanged.

## 5. Evaluation


### Evaluation split

The supervised models are trained using the earlier monthly pairs:

* January → February 2026
* February → March 2026
* March → April 2026
* April → May 2026

The **May → June 2026 pair is held out as the future-period test set**.

The same held-out period is used to evaluate the transparent baseline, Logistic Regression, and Decision Tree. This keeps the comparison on the same evaluation data.

### Evaluation target

The evaluation target is whether a page's CTR declines from the current month to the following month.

The model rankings are therefore evaluated according to how well they place pages with an observed CTR decline near the top of the review queue.

### Metrics

The analysis uses three ranking-oriented metrics:

* **Precision@20** — proportion of target-positive pages among the top 20 ranked pages.
* **Precision@50** — proportion of target-positive pages among the top 50 ranked pages.
* **Average Precision** — summarizes ranking performance across the full held-out test set.

These metrics are calculated separately for the transparent baseline, Logistic Regression, and Decision Tree using their respective ranking scores.

### Held-out results

On the held-out **May → June 2026** test period, the notebook's model comparison evaluates all three approaches using the same three metrics.

The Logistic Regression ranking produced the highest reported Precision@20 and Average Precision, while the Decision Tree produced the highest reported Precision@50. At K=20, 16 of the top 20 Logistic Regression-ranked pages experienced the target CTR decline, while 6 of the top 20 pages selected by the baseline experienced the target decline.

The results indicate that, on this held-out period, the Logistic Regression ranking placed more observed CTR declines near the top of the review queue than the transparent baseline.

### Error interpretation

A ranking error means that a page was prioritized higher or lower than would have been ideal under the defined target.

A false positive in this setting means a page receives a high review priority but does not experience the defined next-month CTR decline. A false negative means a page that does experience the decline is not placed sufficiently high in the review queue.

These errors do not indicate whether a page actually needs a content change. The target measures an observed CTR change, while the final content decision requires human review and additional context.

### Evaluation limits

The evaluation represents **one held-out future period: May → June 2026**.

Therefore, these results should not be interpreted as a universal estimate of future model performance. Additional temporal evaluation across multiple future periods would be needed to determine how consistently the ranking generalizes.

## 6. Interpretation


The analysis found that the **Logistic Regression ranking** placed more pages with an observed next-month CTR decline near the top of the held-out review queue than the transparent baseline on the May → June 2026 evaluation period.

The model uses five observable current-month signals:

* Impressions
* Clicks
* Current CTR
* Average position
* Position bucket

The model combines these signals rather than relying only on the CTR-versus-position benchmark used by the baseline.

### Model contribution interpretation

The Logistic Regression model's feature contributions are calculated from the transformed feature values and learned coefficients.

The strongest positive contribution for each page is converted into a reviewer-facing reason code. A second positive contribution is also retained when available.

The notebook maps these contributions to signals such as:

* `MODEL_SIGNAL_IMPRESSIONS`
* `MODEL_SIGNAL_CLICKS`
* `MODEL_SIGNAL_CURRENT_CTR`
* `MODEL_SIGNAL_AVG_POSITION`
* `MODEL_SIGNAL_POSITION_TOP3`
* `MODEL_SIGNAL_POSITION_4_10`
* `MODEL_SIGNAL_POSITION_11_20`
* `MODEL_SIGNAL_POSITION_20_PLUS`

These codes identify which observed signals contributed most strongly to the model's score for an individual page.

They should **not** be interpreted as explanations of why the page's CTR changed. They describe the model's learned associations with the target.

### What the ranking means

A higher Logistic Regression score means that the page receives a higher predicted probability for the defined target: an observed CTR decline in the following month.

Therefore, pages near the top of the queue are prioritized for human review under this evaluation design.

The score does **not** mean that:

* the page definitely has a content problem,
* the page definitely needs a refresh,
* a particular SEO action should be taken, or
* changing the page will improve its future performance.

### Negative results and uncertainty

The target captures an observed month-to-month CTR decline. It does not capture the reason for that decline or whether a content intervention would reverse it.

The analysis also does not include all page-level context that a human reviewer may need, such as detailed content quality, search-intent interpretation, or business context.

Consequently, the model should be interpreted as a **review-prioritization signal**, rather than a diagnosis or automated recommendation system.

The single held-out May → June evaluation period also limits how confidently the observed ranking performance can be generalized to other periods or data distributions.

## 7. Recommendation



The final output is a ranked review queue based on the **Logistic Regression score**. The ranking is intended to help FlyRank reviewers decide which pages should be inspected first when review time is limited.

A higher score means that the model assigns a higher estimated probability to the defined target: an observed CTR decline in the following month.

### Reviewer workflow

A reviewer can use the queue as follows:

1. Start with the highest-ranked pages.
2. Review the page's current search-performance signals.
3. Inspect the model-derived reason codes to understand which observable signals contributed most to the score.
4. Add page-level content, business, and search-intent context that is not available in the warehouse.
5. Decide whether the appropriate response is refresh, expansion, protection, pruning, monitoring, or no action.
6. Record the human decision separately from the model score.

### Review directions

The ranking can support different review directions depending on the page's context:

| Observed situation                                          | Review direction                                                              |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| High review score with meaningful search visibility         | Inspect the page for possible CTR or content opportunities                    |
| Low CTR relative to its position context                    | Review title, snippet, relevance, and search-intent alignment                 |
| Low visibility or position above 20                         | Review discoverability, relevance, internal linking, and continued investment |
| Strong current performance with elevated decline-risk score | Consider protection or monitoring rather than assuming a major content change |

These are **review prompts, not automatic actions**.

### Confidence and limitations

The recommendation is based on the model's performance on the held-out **May → June 2026** period. Therefore, the ranking should be treated as a prioritization signal rather than a guaranteed prediction of future page performance.

The model also does not determine whether a content change will improve future search performance. Any refresh, expansion, protection, pruning, or monitoring decision requires human review and additional context.

The recommended use of the system is therefore:

> **Use the model to decide which pages to inspect first; use human judgment to decide what, if anything, should be done.**

## 8. Reproducibility


The complete capstone analysis is contained in the capstone notebook committed under:

`work/notebooks/capstone.ipynb`

The notebook is designed to be executed from top to bottom in Google Colab.

### Data access

The notebook reads the FlyRank warehouse release directly from Hugging Face using DuckDB and the `hf://` filesystem.

The source table is:

`fact_content_daily_performance`

The notebook uses monthly partitions from January through June 2026.

Access to the gated dataset requires a Hugging Face access token. The token is retrieved from the Colab Secrets mechanism and is not hard-coded in the notebook.

### Analysis sequence

The notebook reproduces the analysis in the following order:

1. Connect to DuckDB.
2. Authenticate to the gated Hugging Face dataset.
3. Read the required monthly Parquet partitions.
4. Aggregate daily records to the client/content-page level.
5. Construct the monthly temporal pairs.
6. Construct the CTR-decline target.
7. Build the transparent baseline.
8. Create the temporal training and held-out test split.
9. Train Logistic Regression and Decision Tree models.
10. Generate held-out ranking scores.
11. Calculate Precision@20, Precision@50, and Average Precision.
12. Generate the Top-K comparison.
13. Generate the final reviewer queue and model-contribution reason codes.

### Fixed analysis settings

The temporal split is fixed as:

* Training: January → February through April → May
* Held-out test: May → June

The supervised feature list is:

* `impressions`
* `clicks`
* `current_ctr`
* `avg_position`
* `position_bucket`

The Logistic Regression model uses:

* `StandardScaler` for numerical features
* `OneHotEncoder(handle_unknown="ignore")` for `position_bucket`
* `LogisticRegression(max_iter=1000)`

The Decision Tree uses:

* `max_depth=3`
* `random_state=42`

The notebook does not set a separate random seed for Logistic Regression.

### Re-running the notebook

From a fresh Colab session:

1. Open the capstone notebook.
2. Provide the required Hugging Face token through Colab Secrets using the expected `HF_TOKEN` secret.
3. Run the notebook from top to bottom.
4. Confirm that all cells execute without errors.
5. Confirm that the model comparison, Top-K comparison, and final reviewer queue are regenerated.

The notebook itself is the reproducible source for the reported analysis. The final paper should use numbers generated by the current clean run rather than manually entered values from an earlier run.

### Environment note

The completed notebook does not include a committed `requirements.txt` or `pip freeze` environment snapshot. Therefore, exact package-version reproduction is not claimed here.

The analysis relies on the Python/Colab environment and libraries imported by the notebook, including DuckDB, pandas, NumPy, and scikit-learn.

## 9. Acknowledgments & data credit

This project was built using the **FlyRank ML Internship dataset**.

Data source: [FlyRank](https://flyrank.ai)
