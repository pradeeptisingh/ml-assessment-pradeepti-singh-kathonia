# Part B — Business Case Analysis
## Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### (a) ML Problem Formulation

**Target Variable:**
`items_sold` — the number of items sold at a given store in a given month under a specific promotion.

**Candidate Input Features:**

| Feature Group | Features |
|---|---|
| Store characteristics | store_id, store_size, location_type, monthly_footfall, competition_density |
| Promotion | promotion_type (one-hot encoded) |
| Temporal | month, year, is_weekend, is_festival, is_month_end |
| Customer demographics | avg_customer_age_band, gender_split, loyalty_member_pct (if available) |
| Historical performance | avg_items_sold_last_3_months, promotion_response_history per store |

**Type of ML Problem:**
This is a **supervised regression problem**. The target variable (`items_sold`) is continuous and numeric, making regression the appropriate framing.

Nevertheless, the *goal of the business problem*, which is to choose the best promotion in each store each month, involves solving what is known as a **recommendation problem** that has a combinatorial nature. One possible solution is to construct a regression model that outputs items_sold for each (store, promotion) pair and select the one with the maximum expected items_sold.

**Justification:**
- The variable being predicted is a continuous variable, not a category, and hence, classification cannot be applied.
- The algorithm must be able to prioritize promotions according to their outcomes, which necessitates regression values.
- Moreover, regression allows for understanding the magnitude of difference between two promotions, for example, "Promotion A will result in 42 more sales than Promotion B."

---

### (b) Items Sold vs Revenue as Target Variable

**Why `items_sold` is more reliable than revenue:**

1. **Promotion confounding on revenue:** The Flat Discount and BOGO have an immediate impact on the income earned from each item. It is possible for a promotion to have a high amount of revenue because it simply charges a high price, even when its performance is low.

2. **Promotion goal alignment:** The goal of the marketing group is to influence *consumer behavior* — to make people purchase more goods. The number of products sold directly reflects this behavior. Income is an output variable that depends on pricing actions which lie beyond the model’s realm.

3. **Consistency across store sizes:** The urban location is always going to make a lot more money than the rural location. Normalising by using sales makes comparison between stores possible.

4. **Simpler causal chain:** The promotional effect on sales is more direct than its effect on revenues, which is further moderated by the price, discounts, and contents of the purchase basket.

**Broader principle — Target Variable Selection:**
This serves as an example of the **fundamental rule of measuring the variable we want to impact and not something else down the line.** In practice, ML targets tend to be selected for their ease of measurement (revenue) and not for capturing the causal factor we actually care about (purchase behavior). An improperly selected target will lead to a model that accurately predicts the wrong thing, a concept known as **Goodhart’s law**, which states that once a metric is used as a target, it stops being a good metric.

---

### (c) Single Global Model vs Location-Stratified Strategy

**Problem with a single global model:**
A global approach to modeling averages the response function for all the stores. Should the customers in the urban stores react totally differently to the same promotion compared to those in the rural stores, then the model would capture an averaged and hence incorrect response function.

**Proposed alternative: Stratified or Hierarchical Modelling**

| Strategy | Description | Best For |
|---|---|---|
| **Separate models per location type** | Train three independent models (urban, semi-urban, rural) | Moderate data volume; clean segment boundaries |
| **Store-level fixed effects** | Include `store_id` as a one-hot feature in a single model | Sufficient data per store (≥24 months) |
| **Hierarchical (mixed-effects) model** | Partial pooling: global trends + store-level deviations | Small per-store data; best generalisation |
| **Clustered ensemble** | Cluster stores by behaviour profile, one model per cluster | When location type alone is insufficient |

**Recommended approach:** A **two-level strategy** — global model training using `location_type`, `store_size`, and `store_id` embeddings as features, but including **residual corrections at the store level** based on the history of how that store responded to past promotions. This takes into account both the global trends as well as the peculiarities of individual stores without resorting to separate models.

**Justification:** The stores in various areas vary according to the volume of traffic, demographic profile, and competition. Promotion strategies that lead to self-cannibalization for an urban, competitive environment may be very successful in a rural environment with low competition. This is bound to give rise to systematic bias.

---

## B2. Data and EDA Strategy

### (a) Table Joins and Data Grain

**Four source tables:**

| Table | Key Columns | Grain |
|---|---|---|
| transactions | store_id, date, items_sold, promotion_i | One row per transaction / per store-day |
| store_attributes | store_id, store_size, location_type, footfall, competition_density | One row per store (slowly changing) |
| promotion_details | promotion_id, promotion_type, discount_depth | One row per promotion |
| calendar | date, is_weekend, is_festival, month, year | One row per calendar date |

**Join strategy (left joins from transactions as the base):**

```
transactions
  LEFT JOIN store_attributes   ON transactions.store_id    = store_attributes.store_id
  LEFT JOIN promotion_details  ON transactions.promotion_id = promotion_details.promotion_id
  LEFT JOIN calendar           ON transactions.date         = calendar.date
```

**Final dataset grain:**
**One row = one store × one month × one promotion type.**

This means aggregating daily transactions up to the monthly store-promotion level before modelling, since the business decision is made monthly.

**Aggregations performed before modelling:**

| Aggregation | Method |
|---|---|
| `items_sold` (target) | `SUM` of daily items sold per store-month-promotion |
| `footfall` | `SUM` or `AVG` per store-month |
| `is_festival` | `MAX` (1 if any festival day in the month) |
| `is_weekend_days` | `COUNT` of weekend days in the month |
| `promotion_active_days` | `COUNT` of days the promotion ran |
| Historical features | Rolling 3-month avg items sold per store (lag feature) |

**Important:** The rows for months where there was no promotion by the store remain in the database with `promotion_type = 'none',` since survivorship bias should be avoided; the model needs to know about the baseline performance without the promotion.

---

### (b) EDA Strategy

**Analysis 1 — Target Variable Distribution (Histogram + Boxplot)**
*What to look for:* Is the variable `items_sold` normally distributed, right-skewed, or bimodal? Are there any outliers (such as stores where sales were very high due to festival seasons)?
*Impact on modeling:* In case of skewness, the target will need to be logged. Outliers in festival seasons are valid points and must not be dropped.

**Analysis 2 — Promotion Effectiveness by Location Type (Bar Chart)**
*What to look for:* Which promotion is associated with the maximum mean of `items_sold` among urban, semi-urban, and rural shops? Is the ranking pattern similar or different for each location category?
*Modeling the interaction effect:* If the pattern varies depending on location categories (interaction effect), an interaction term (`promotion_type x location_type`) should be included in the model.

**Analysis 3 — Correlation Heatmap (Numeric Features)**
*What to look for:* Strong positive correlation between `competition_density` and `items_sold`; strong positive correlation between `footfall` and `items_sold`; presence of multicollinearity among features (for instance, `store_size` and `footfall` could be positively correlated, resulting in redundant features).
*Effect on model construction:* High correlations between two features should prompt feature selection or dimensionality reduction. Tree-based models are insensitive to multicollinearity, but it creates larger coefficient variances in linear models.

**Analysis 4 — Temporal Trends (Line Plot: monthly avg items_sold over 3 years)**
*What to look for:* Seasonal variation (e.g., peak sales during November to December, low sales during January to February)? Trend from year to year (increase/decrease)? Breaks in the structural relationship (e.g., the month when all the stores have a dip, suggesting either poor-quality data or some external events)?
*Modeling Effect:* Any pattern of seasonal variation will mean that `month` is included as a cyclic variable (using sinusoidal transformation instead of ordinal integer), or introducing lag variables.

**Analysis 5 — Promotion Distribution Over Time (Stacked Bar per Month)**
*What to look for:* Was there equal use of each promotional campaign over each month, or were the Flat Discount promotions only employed during sales that occurred in January? If particular promotions are confounded with particular months, it will not be possible to separate out the effect of the promotion from the effect of seasonality.
*Impact on Modelling:* Confounded Promotions would need causal inference methods such as propensity score weighting or, at the very least, a robust seasonal covariate to be de-confounded.

---

### (c) Promotion Imbalance (80% No-Promotion Transactions)

**How imbalance affects the model:**
The regression model, trained on an imbalanced dataset, will be trained predominantly based on cases of no promotion. Thus, it will output highly accurate probabilities for the case of “no promotion,” whereas for the other two cases, it is expected to generate badly calibrated probabilities since there are not enough examples of such. This leads to underestimation for rarer promotions, such as Free Gift, and overestimation for others.

**Steps to address it:**

1. **Re-frame the target as promotion uplift:** Predict the difference between `items_sold_during_promotion` and `baseline_items_sold` rather than raw `items_sold`. The latter is the store’s average sales when there is no promotion for that month.

2. **Stratified sampling / oversampling:** In the training phase, oversample promoted transactions in order to make sure that all promotion types have enough samples. Alternatively, the promoted transactions can be upscaled using the `sample_weight` parameter within scikit-learn.

3. **Separate baseline model:** Two models must be built; one for predicting baseline items sold (i.e., no promotion applied), and another model for predicting the uplift. The two models' outputs can simply be added together during inference to obtain the uplift prediction, which is a common method used in uplift modeling for businesses.

4. **Report model uncertainty by promotion type:** Use confidence intervals to warn when predictions for infrequently observed promotions are unreliable.

---

## B3. Model Evaluation and Deployment

### (a) Train-Test Split Strategy and Evaluation Metrics

**Train-test split design:**
Given three years of data across 50 stores, the correct approach is a **temporal split**:

- **Training set:** Months 1–30 (first 2.5 years, ~83% of data)
- **Test set:** Months 31–36 (final 6 months, ~17% of data)

For more robust evaluation, use **time-series cross-validation (walk-forward validation)**:
- Fold 1: Train on months 1–12, test on months 13–15
- Fold 2: Train on months 1–15, test on months 16–18
- Fold 3: … and so on, expanding the training window each time.

This tests the model's ability to generalise to genuinely unseen *future* data at each fold.

**Why random split is inappropriate:**
Using random splits for training the model using records in month 36 and testing it on month 6 enables the model to learn future seasonality and future store behavior, resulting in an inflated test score. However, in reality, no such future data would be available when deploying the model.

**Evaluation metrics:**

| Metric | Formula | Interpretation in this context |
|---|---|---|
| **RMSE** | √(mean of squared errors) | Penalises large errors heavily. A RMSE of 30 means predictions are typically off by ±30 items. Sensitive to outlier months (festivals). |
| **MAE** | mean of absolute errors | More interpretable: "on average, we are off by X items per store-month." Less sensitive to festival outliers. Report to marketing team. |
| **MAPE** | mean of |error|/actual | Percentage error — useful for comparing stores of different sizes. Avoid if any stores have near-zero items sold (division issue). |
| **R²** | 1 − SS_res/SS_tot | Proportion of variance explained. R² > 0.75 indicates a useful model. Compare against a naive baseline (predicting the store's historical mean). |

**Primary metric recommendation:** Use **MAE** as the headline metric for business communication (intuitive units) and **RMSE** as the technical metric for model selection (penalises costly large misses).

---

### (b) Feature Importance for Communicating Recommendations

**Why the model recommends differently for Store 12 in December vs March:**

The recommendation algorithm calculates the number of `items_sold` for each promotion of the five promotions offered by Store 12 during that particular month. The different recommendations come about since there is a variation in the input feature vector between months.

**Investigation steps:**

1. **Extract predictions for all 5 promotions for Store 12 in both months:**
   Create a table with the predicted value of `items_sold` per promotion for both December and March. This highlights the recommendation made and the level of confidence that goes with it.

2. **Identify which features changed between months:**
  For Store 12, the predictors `month`, `is_festival`, `is_weekend_days`, and any rolling average predictors will vary from December to March. Determine which predictors are responsible for the change in the recommendation due to their varying values during the two periods.


3. **Use SHAP (SHapley Additive exPlanations) values:**
   SHAP allows for a breakdown of the effect of each feature in contributing to a specific outcome. Produce SHAP waterfalls for Store 12 during December and March.

   *Example output:*
   - December: `is_festival=1` (+42 items), `month=12` (+31 items), `promotion_type=loyalty_points` (+18 items)
   - March: `is_festival=0` (0 items), `month=3` (−12 items), `promotion_type=flat_discount` (+25 items)

4. **Communicate to the marketing team using plain language:**
   >"Based on the analysis in December, the model suggests a Loyalty Point offer, since Store 12 customers will be in a mood of buying gifts, and will want to collect the points for their future consumption. 
   >Festival flag is the largest factor leading to this decision.As for March, when the festive period is over, people are less inclined towards shopping, which makes them sensitive to prices. Flat discount serves as the best alternative in this scenario."

   This narrative should always be grounded in the SHAP values and validated against historical data from Store 12.

---

### (c) End-to-End Deployment Process

**Step 1 — Model Serialisation (Saving the Model)**

After training on the full historical dataset, save the entire scikit-learn `Pipeline` object (which includes the `ColumnTransformer` preprocessor and the trained model) using `joblib`:

```python
import joblib
joblib.dump(pipeline, 'promotion_recommender_v1.pkl')
```

Saving the full pipeline is critical — it ensures the *same* preprocessing (OHE vocabulary, scaler mean/std) is applied at inference time as at training time. Never save only the model weights without the preprocessor.

Version the saved model file (e.g. `v1`, `v2`) and store it in a model registry (MLflow, AWS S3, or a versioned file store) alongside the training metadata: training date range, feature list, evaluation metrics.

**Step 2 — Monthly Inference Pipeline**

At the start of each month, an automated pipeline runs the following steps:

```
1. DATA EXTRACTION
   Pull the latest store attributes, promotion calendar, and festival flags
   for the upcoming month from the data warehouse.

2. FEATURE CONSTRUCTION
   For each of the 50 stores × 5 promotions = 250 (store, promotion) combinations:
   - Build the feature vector: store features + promotion_type + month features
   - Compute rolling lag features from the most recent 3 months of actual sales.

3. LOAD MODEL
   Load promotion_recommender_v1.pkl from the model registry.

4. GENERATE PREDICTIONS
   model.predict(X_new) for all 250 rows.

5. SELECT RECOMMENDATION
   For each store, return the promotion type with the highest predicted items_sold.

6. OUTPUT
   Write recommendations to a dashboard or email report for the marketing team.
   Include the predicted items_sold and a confidence band for each recommendation.
```

**Step 3 — Monitoring and Retraining Triggers**

After each month, once actuals are available, the monitoring system computes:

| Monitor | Signal | Threshold / Action |
|---|---|---|
| **Prediction error tracking** | Monthly MAE and RMSE on actuals vs predictions | Alert if MAE exceeds 1.5× training MAE |
| **Data drift detection** | Track distribution of input features (e.g. footfall, competition density) using Population Stability Index (PSI) | PSI > 0.2 → flag feature for investigation |
| **Concept drift detection** | Monitor if the relationship between promotions and sales has shifted (e.g. a new competitor enters the market) | Use a sliding window model trained on recent 6 months; compare its predictions to the deployed model's predictions |
| **Recommendation diversity** | Check if the model is recommending the same promotion for all stores | Flag if any single promotion accounts for > 60% of recommendations |
| **Business feedback loop** | Marketing team flags when they override a recommendation and track override rate | Override rate > 20% → schedule a model review meeting |

**Retraining schedule:**
- **Scheduled retraining:** Every 6 months, retrain on the full updated dataset.
- **Triggered retraining:** If MAE surpasses threshold value or any drift in PSI is identified then.
- It is necessary to evaluate the updated model with respect to a test period in time; do not deploy a new model until it is tested against the existing one.
