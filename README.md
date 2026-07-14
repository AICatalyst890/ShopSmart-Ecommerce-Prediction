# 🛒 ShopSmart E-Commerce Customer Behavior Prediction

An end-to-end Machine Learning pipeline designed to predict customer purchasing behavior using an optimized *Decision Tree Classifier*. This project tackles the real-world challenge of *algorithmic overfitting* by moving from automated, split-localized grid searches to disciplined, structural *pre-pruning techniques*.

---

## 📊 Project Pipeline & Architecture

The pipeline is built with a modular, leakage-free `scikit-learn` framework to maintain strict production data hygiene.

```text
[ shop_smart_ecommerce.csv ] 
              │
              ▼
     [ 80/20 Stratified Split ] ───► Prevents Data Leakage
              │
              ├─► Numerical Features ───► [ StandardScaler ]
              │
              └─► Categorical Features ─► [ OneHotEncoder ]
              │
              ▼
     [ ColumnTransformer ] ────► Merges Feature Layers
              │
              ▼
 [ DecisionTreeClassifier ] ───► Structurally Pre-Pruned

```

### 1. Data Splitting

* **Strategy:** An 80/20 train-test split is executed *before* any feature transformations are applied to avoid processing or target leaks.
* **Stratification:** Stratified by the target variable (`Revenue`) to perfectly maintain original class proportions across both training and testing subsets.

### 2. Isolated Preprocessing (`ColumnTransformer`)

* **Numerical Path:** Features reflecting continuous user engagement and duration telemetry (`Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`) are normalized using `StandardScaler` to ensure scale stability.
* **Categorical Path:** Features tracking environment configurations (`OperatingSystems`, `Browser`, `Region`, `TrafficType`) are encoded via `OneHotEncoder` to handle discrete multi-class arrays cleanly without introducing artificial ordinal bias.

---

## ⚙️ Technical Highlights

* **Target Optimization:** Raw target values are cast explicitly to `int64` ($0$ and $1$) to provide a stable, standard data type for categorical metric logging and confusion matrix plotting.
* **Leakage Prevention:** By wrapping transformations inside an explicit `Pipeline`, statistics from `StandardScaler` and category mappings from `OneHotEncoder` are computed strictly on the training fold, ensuring zero distribution leakage from the held-out validation set.
* **Structural Interpretability:** The final model is fully visualized down to a constrained maximum depth, ensuring that every split criteria can be traced back to core behavioral features like information gain metrics.

---
## 📈 Verified Model Performance Metrics Before vs. After Pre-Pruning

The following metrics contrast the initial automated `GridSearchCV` baseline model against the final, manually constrained pre-pruned model across your 80/20 stratified splits:

| Evaluation Phase | Model State | Training Score | Testing Score | Diagnostic & Operational Notes |
| :--- | :--- | :---: | :---: | :--- |
| **F1-Score (Macro)** | Before Pre-Pruning (`GridSearchCV`) <br> **After Pre-Pruning (Cell [15])** | — <br> **0.6812** | **0.5984** <br> **0.6629** | The automated search overfitted to local noise, causing a metric collapse. Pre-pruning closed the gap ($\Delta < 2\%$). |
| **Overall Accuracy** | Before Pre-Pruning <br> **After Pre-Pruning** | — <br> **91%** | — <br> **90%** | High baseline correctness maintained across both customer segments. |
| **Precision (Class 1 - Buyers)** | Before Pre-Pruning <br> **After Pre-Pruning** | — <br> **74%** | — <br> **72%** | Highly reliable positive class targeting accuracy for live campaigns. |
| **Recall (Class 1 - Buyers)** | Before Pre-Pruning <br> **After Pre-Pruning** | — <br> **63%** | — <br> **62%** | Stable extraction capture rate for isolating true conversion events. |

### Training Confusion Matrix Diagnostic (After Pre-Pruning)
$$
\begin{bmatrix} 
7990 & 348 \\ 
558 & 968 
\end{bmatrix}
$$

---

## 🛠️ Overfitting Mitigation Theory

A common failure mode when training decision trees on high-dimensional behavioral datasets is **overfitting**—where an unconstrained model continues branching until it perfectly memorizes the training data's noise, causing validation metrics to collapse.

### Automated Grid Search vs. Manual Pre-Pruning

* **The Automated Dilemma:** Standard hyperparameter grid searches (`GridSearchCV`) often prioritize maximizing localized validation split adjustments. This can select configurations that chase statistical anomalies within specific folds rather than parsing broad behavioral trends.
* **The Engineering Fix:** This pipeline enforces rigid, manual **structural pre-pruning** limits directly on the tree's growth. By constraining parameters such as `max_depth=5` and `min_samples_leaf=80`, the model is forced to generalize across broad categorical paths rather than carving out hyper-specific, fragile leaf clusters that fail on unseen data.

---

## 💼 Strategic Business Application Value

Deploying a model with a **72% Testing Precision** for buyers translates directly into high-impact conversion optimization for e-commerce platforms:

* **Ad Spend Efficiency:** Marketing teams can confidently shift from broad, expensive remarketing campaigns to precision-targeted promotions. Delivering ad spend exclusively to users with a high predicted intent reduces budget waste on non-converting traffic.
* **User Experience Personalization:** Real-time intent predictions enable dynamic platform changes—such as highlighting limited-time discounts or offering live chat assistance specifically to high-probability buyers before they drop off the conversion funnel.

---

## 🚀 Quick Setup & Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/AICatalyst890/ShopSmart-Ecommerce-Prediction.git
cd ShopSmart-Ecommerce-Prediction

```

2. **Install core pipeline dependencies:**
```bash
pip install -r requirements.txt

```

3. **Launch the development environment:**
```bash
jupyter lab ShopSmartEcommerce.ipynb

```

---

## 📂 Repository Structure

* `ShopSmartEcommerce.ipynb` — Comprehensive notebook detailing preprocessing, baseline evaluations, structural visualizations, and tree pruning logs.
* `shop_smart_ecommerce.csv` — Anonymized session telemetry data containing customer browsing history and purchase flags.
* `requirements.txt` — Environment manifest defining package versions (`scikit-learn`, `pandas`, `numpy`, `matplotlib`).
* `README.md` — Project architecture, optimization metrics, and mitigation documentation.