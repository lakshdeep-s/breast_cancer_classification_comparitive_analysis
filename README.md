<style>
:root {
    --primary: #a826e4ff;
    --primary-100: #7512a2ff;
    --ideal: #06a21bff;
    --keyword: #d56209ff;
    --muted: #5f5f5fff;
}
.badge {
    background-color: var(--primary-100);
    font-size: 0.875rem;
    padding: 5px 10px;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.25px;
}
.keyword {
    color: var(--keyword);
    font-style: italic;
}
.ideal {
    color: var(--ideal);
}
.small-caption {
    font-size: 0.875rem;
    color: var(--muted);
}
.section-badges {
    display: flex;
    gap: 1rem;
}
h1, h2, h3 {
    color: var(--primary);
}
</style>

<!-- Title and description -->
# Breast Cancer ML+DL: Malignancy-First Comparative Study 📈
A research-style, reproducible pipeline comparing traditional ML models and an MLP on the Wisconsin Breast Cancer dataset, engineered to maximize malignant recall and reduce dataset dimensionality without sacrificing overall performance.

<!-- Abstract / Executive Summary -->
## Executive Summary

This project presents a comprehensive pipeline for breast cancer classification on the Wisconsin dataset, emphasizing a malignancy-first approach. By integrating rigorous exploratory data analysis, feature engineering, and advanced modeling—including Random Forest, SVM, XGBoost, and Neural Networks—the pipeline prioritizes maximizing malignant recall while maintaining high overall accuracy. A key innovation lies in aggressive dimensionality reduction, distilling over 30 features into a concise, interpretable subset of approximately 7–10 critical predictors. The study leverages stratified cross-validation, consistent scoring metrics focused on malignant detection, and configuration-driven reproducibility to ensure robustness and transparency across modeling approaches. The resulting models demonstrate strong performance, achieving recall rates near or above 95% on withheld test data, making them suitable for real-world clinical screening scenarios.

<div class="keyword">Keywords: malignant recall, feature engineering, stratified cross-validation, dimensionality reduction, interpretable ML, reproducible pipeline</div>

## Project Motivation & Impact

<div class="section-badges">
  <span class="badge">Malignant Recall</span>
  <span class="badge">Feature Selection</span>
  <span class="badge">Clinical Interpretability</span>
  <span class="badge">Balanced Performance</span>
  <span class="badge">Reproducible Pipeline</span>
</div>

<br>

The driving force behind this project lies in the critical clinical need to accurately identify malignant breast cancer cases with high sensitivity. The key motivations and impacts include:

- <span class="ideal">**Clinical Priority:**</span> Missing a malignancy (false negative) carries far graver consequences than false alarms, making malignant recall the paramount metric in cancer screening and diagnosis.

- <span class="ideal">**Goal Orientation:**</span> The primary goal was to boost recall for the malignancy class to reduce missed true positives, while simultaneously increasing overall accuracy and robustness in both training and inference phases.

- <span class="ideal">**Feature Optimization:**</span> Through a rigorous feature selection strategy, the project reduced the dataset dimensionality from 30+ features to a concise, interpretable set of approximately 7–10 key inputs, facilitating efficiency and enhancing clinical trust.

- <span class="ideal">**Balanced Trade-offs:**</span> Despite aggressive dimensionality reduction, the pipeline was able to maintain high overall accuracy and balanced classification metrics, demonstrating that performance gains and parsimony can coexist.

- <span class="ideal">**Scientific Rigor & Reproducibility:**</span> The end-to-end workflow is config-driven, reproducible, and grounded in domain expertise, combining robust feature engineering, stratified evaluation, and malignant-focused tuning for reliability and transparency.

This study uniquely merges domain knowledge with methodical machine learning to address a high-stakes clinical problem, providing a practical, interpretable, and high-performing solution for breast cancer malignancy classification.

## Dataset and Problem Framing

<div class="section-badges">
  <span class="badge">Wisconsin Dataset</span>
  <span class="badge">High-Dimensional</span>
  <span class="badge">Feature Correlation</span>
  <span class="badge">Class Imbalance</span>
  <span class="badge">Clinical Constraints</span>
</div>
<br>
The Wisconsin Breast Cancer dataset is a well-established benchmark for binary classification in medical diagnostics. Despite its relatively small size of 569 samples, it serves as a valuable foundation for building and evaluating classification pipelines due to its quality and clinical relevance.

- <span class="ideal">Dataset Characteristics:</span> Contains 30 continuous features capturing cell nucleus properties like radius, texture, perimeter, area, smoothness, symmetry, and fractal dimension, each evaluated as mean, standard error, and worst-case values. This results in a high-dimensional dataset that challenges modeling efforts.

- <span class="ideal">Feature Correlation:</span> Many features are strongly correlated, as confirmed by detailed exploratory data analysis. This redundancy motivates robust feature selection to reduce dimensionality, avoid overfitting, and improve interpretability.

- <span class="ideal">Class Distribution:</span> The dataset exhibits a mild class imbalance, with approximately 37% malignant and 63% benign cases, necessitating consideration in model training and evaluation.

- <span class="ideal">Clinical Imperatives:</span> Missing a malignant case (false negative) is far more consequential than a false positive. Therefore, malignant recall (sensitivity) is the highest priority metric, shaping model optimization and validation strategies.

- <span class="ideal">Evaluation Strategy:</span> Use of stratified cross-validation preserves class proportions in training and test splits, ensuring robust and fair assessments. Metrics like precision, recall, and F1-score complement overall accuracy to provide nuanced performance insights.

These aspects collectively inform the design of a pipeline that prioritizes clinical safety and statistical rigor through tailored preprocessing, feature engineering, and model evaluation.

## Exploratory Data Analysis (EDA)

<div class="section-badges">
  <span class="badge">Outliers</span>
  <span class="badge">Correlation</span>
  <span class="badge">Dimensionality Reduction</span>
</div>

The exploratory data analysis (EDA) phase provided critical insights into the Wisconsin Breast Cancer dataset’s structure and informed subsequent feature engineering and modeling steps.

### Outlier Analysis

An initial examination of feature distributions revealed the presence of meaningful outliers across multiple variables. Box plots highlight that while some features exhibit typical Gaussian-like spreads, others present heavy tails indicative of extreme values, which are clinically relevant and thus retained rather than removed. These outliers reflect biological variability and are essential for robust classification.

![Box Plot of Raw Features](images/outliers_boxplot.png)  
*Figure 1: Box plots of original features showing distribution spread and outliers.*

### Correlation Insights

Calculation of Pearson's rank correlation coefficients across all features exposed significant multicollinearity, with clear grouping of related measurements such as radius, perimeter, and area forming clustered blocks. This strong correlation suggests redundancy in raw features, motivating dimensionality reduction to eliminate noise and enhance model interpretability.

![Pearson Correlation Heatmap](images/correlation_heatmap.png)  
*Figure 2: Pearson correlation matrix displaying strong feature clusters in the original dataset.*

### Dimensionality Reduction Visualizations

To visualize the separability of classes in the high-dimensional space, t-SNE projections were generated:

- **Original Dataset (All Features):** The t-SNE plot illustrates overall class separation, though several malignant samples overlap with benign clusters, indicating borderline cases.

![t-SNE Original Features](images/tsne_original.png)  
*Figure 3: t-SNE visualization of the full original feature set.*

- **Engineered Feature Pool (Top 20 Features):** Using the selected engineered features, t-SNE reveals tighter clustering and clearer separation between malignant and benign classes, signaling enhanced discriminative power post-engineering.

![t-SNE Engineered Features](images/tsne_engineered.png)  
*Figure 4: t-SNE plot based on the engineered feature subset.*

- **Raw Feature Pool (Top 20 Features):** For comparison, the raw feature subset (excluding engineered features) shows less distinct clusters and greater class overlap, underscoring the value added by feature engineering.

![t-SNE Raw Features](images/tsne_raw.png)  
*Figure 5: t-SNE plot based on the raw feature subset.*

---

Together, these analyses confirm the importance of targeted feature engineering and selection to reduce dimensionality, mitigate redundancy, and improve class separability, thereby setting the stage for effective model training and evaluation.
