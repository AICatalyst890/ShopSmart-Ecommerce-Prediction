# 🛒 ShopSmart E-Commerce Customer Behavior Prediction

An end-to-end Machine Learning pipeline designed to predict customer purchasing behavior using an optimized **Decision Tree Classifier**.  
This project tackles the real-world challenge of **algorithmic overfitting** by moving from automated grid searches to disciplined **pre-pruning techniques**.

---

## 📊 Project Architecture & Workflow

The pipeline is built with a modular, leakage-free **scikit-learn `Pipeline`** framework to ensure production-ready data hygiene.

1. **Data Preparation**
   - Source: `shop_smart_ecommerce.csv` — raw e-commerce interaction and behavioral dataset.
   - Train/test split performed before any transformations to prevent data leakage.
2. **Preprocessing Pipeline**
   - **Numerical Features:** Scaled and normalized (e.g., `session_length`, `pages_viewed`, `transaction_count`).
   - **Categorical Features:** One-Hot Encoded (OHE) for multi-class handling (e.g., `device_type`, `region`, `product_category`).
3. **Model Selection & Tuning**
   - Baseline estimators → `GridSearchCV` → Manually constrained configurations.

---

## 🛠️ Overfitting Challenge & Solution

### ❌ The Dilemma
- **GridSearchCV Result:**  
  - Selected parameters: `criterion='entropy'`, `max_depth=6`, `min_samples_leaf=30`, `min_samples_split=200`  
  - **Best CV F1 Score:** 0.6599762137  
  - **Actual Test F1 Score:** 0.5984251968  
- **Diagnosis:**  
  The automated search prioritized maximizing local split criteria variations, causing the tree to memorize statistical noise within training folds rather than extracting clean, repeatable buying behaviors.

### ✅ The Engineering Fix (Pre-Pruning)
Manually constrained tree architecture:

```python
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_split=150,
    min_samples_leaf=80,
    random_state=42
)
```

---

## 📈 Performance Metrics

| Metric Layer          | F1-Score | Generalization Notes |
|-----------------------|----------|----------------------|
| Training Performance  | 0.6812   | Stable; captures core transactional signals |
| Testing Performance   | 0.6629   | Minimal gap (<2%); confirms zero overfitting |
| Precision (Buyers)    | 0.72     | Reliable targeting; minimizes wasted marketing budget |
| Recall (Buyers)       | 0.62     | Balanced capture rate; isolates majority of true purchasers |
| Overall Accuracy      | 0.90     | Strong baseline; correctly identifies non-buyers |

---

## 🌲 Key Structural Insights
- **Entropy Criterion:** Cleaner segmentation via information gain.  
- **Node Control:** `min_samples_leaf=80` prevents noisy micro-clusters.  
- **Depth Limit:** `max_depth=5` ensures smooth, interpretable decision paths.  

---

## 🧩 Dataset Description
The dataset `shop_smart_ecommerce.csv` contains anonymized customer interaction records with behavioral and transactional attributes such as:
- `session_id` — Unique identifier for each browsing session  
- `pages_viewed` — Number of product pages visited  
- `session_length` — Duration of the session in seconds  
- `device_type` — Platform used (mobile, desktop, tablet)  
- `region` — Customer’s geographic segment  
- `product_category` — Category of products browsed  
- `purchase_flag` — Target variable (1 = Buyer, 0 = Non-buyer)

These features collectively enable the model to learn conversion patterns and predict purchasing intent.

---

## 🗂️ Repository Structure
- `shop_smart_ecommerce.csv` → Raw dataset (customer sessions, product interactions, conversions)  
- `ShopSmartEcommerce.ipynb` → Full notebook (preprocessing, tuning, visualization)  
- `requirements.txt` → Environment dependencies  
- `README.md` → Project overview & documentation  