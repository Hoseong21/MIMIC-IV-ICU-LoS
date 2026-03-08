# ICU Length of Stay Prediction
### Medication Stabilization Pattern–Based Modeling (MIMIC-IV)

약물 안정화 패턴을 활용한 중환자실(ICU) 체류 기간(LOS) 예측 프로젝트입니다.

<br>

## 📌 Project Info

| **Period** | 2025.09.15 – 2025.12.09 |
|---|---|
| **Team** | 이재용(팀장), 김대건, 송호성, 김지웅 |
| **Course** | 데이터애널리틱스 팀프로젝트 |
| **Data** | MIMIC-IV v2.2 |

<br>

## 📖 Overview

본 프로젝트는 **MIMIC-IV v2.2 ICU 데이터**를 활용하여 중환자실 환자의 **Length of Stay (LOS)** 를 예측하는 머신러닝 기반 분석 프로젝트입니다.

기존 연구들은 주로 Vital sign 및 Laboratory 지표 중심의 예측 모델을 사용해 왔습니다. 본 프로젝트에서는 이를 확장하여 **약물 투여 패턴과 환자의 생리적 안정화 과정**을 함께 고려하는 접근 방식을 설계하였습니다.

<br>

## ✨ Key Contributions

- 생체 신호 안정성과 약물 투여 안정성을 결합한 **Stabilization Index** 를 신규 설계하였습니다.
- Vital, Laboratory, Medication 기록을 통합한 **멀티모달 피처 엔지니어링**을 수행하였습니다.
- Tree-based, Deep Learning, AutoML 기반의 다양한 tabular 모델을 비교 실험하였습니다.
- **AutoGluon MAE 1.5644 / RMSE 2.5657** 을 달성하였으며, 단기 체류(≤4일) 구간에서 기존 연구 대비 우수한 성능을 기록하였습니다.
- SHAP 및 LIME을 활용하여 모델의 예측 근거를 해석하고 임상적 인사이트를 도출하였습니다.

<br>

## 🗂 Dataset

**Database** : MIMIC-IV (v2.2)

**Population Criteria**
- 성인 ICU 환자 (18세 이상)
- 첫 번째 ICU 입실만 포함
- LOS : 2 < LOS ≤ 21 days
- ICU 퇴실 생존 환자만 포함
- Missing 및 out-of-range 데이터 제거
- ICU 입실 후 **48시간 관측 window** 데이터 사용

**Tables Used**

| Table | Description |
|---|---|
| `icustays` | ICU 입퇴실 정보 |
| `patients` | 환자 기본 정보 |
| `admissions` | 입원 정보 |
| `chartevents` | Vital sign 측정 기록 |
| `labevents` | 검사 수치 기록 |
| `inputevents` | 약물 투여 기록 |

**Final Cohort : 16,349 ICU stays**

<br>

## ⚙️ Feature Engineering

### 1. Vital Features

ICU 입실 후 48시간 구간 데이터를 기준으로 요약 통계량(`mean / min / max / std`)을 생성하여 환자의 생리적 상태와 변동성을 모델에 반영하였습니다. GCS 항목은 `mean / min / max` 만 사용하였습니다.

- Heart Rate (심박수)
- Respiratory Rate (호흡수)
- SpO₂ (산소포화도)
- Body Temperature (체온)
- GCS Eye / Verbal / Motor (의식 수준 지표)

<br>

### 2. Lab Features

ICU 입실 후 48시간 구간 데이터를 기준으로 각 항목의 `mean / min / max / std` 를 생성하였습니다. 결측률이 높은 Albumin, Bilirubin은 제외하였습니다.

- Creatinine, BUN (신장 기능)
- WBC (감염 및 염증)
- Platelet Count, Glucose (혈액 기능, 혈당)
- Lactate, AnionGap, Bicarbonate, pH (쇼크 및 산-염기 균형)
- Potassium, Sodium (전해질)
- INR (응고 기능)

<br>

### 3. Medication-Based Features

약물 투여 기록을 기반으로 환자의 치료 안정성, 강도, 복잡도 및 중증도를 반영하는 약물 피처를 설계하였습니다. 약물은 임상적 의미에 따라 **5개 class** (Vasoactive, Analgosedation, Diuretic, Insulin, Other) 로 분류하였으며, 피처는 다음 4개 그룹으로 구성하였습니다.

| Group | Feature | Description |
|---|---|---|
| **Stability** | `overall_rate_cv` | 약물 투여량 변동계수 (CV = std / mean) |
| **Dosage & Exposure** | `analgosedation_intensity` | 진정·진통제 시간당 투여 강도 |
| | `total_sedation_exposure` | 진정제 누적 투여량 |
| | `insulin_requirement` | 인슐린 누적 투여량 |
| | `vasoactive_load` | 혈관작용제 사용 여부 × 투여 시간 |
| **Complexity** | `drug_diversity_index` | 고유 약물 수 / 최대 동시 투여 약물 수 |
| | `drug_class_ratio` | 약물 분류 간 비율 |
| **Severity & Risk** | `critical_medication_score` | 중증 약물 가중치 기반 점수 |
| | `medication_escalation` | 시간에 따른 약물 강도 증가 패턴 |
| | `concurrent_intensity` | 동시 투여 약물 최대 개수 |

<br>

### 4. Stabilization Index ⭐

생체 신호 안정성과 약물 투여 안정성을 결합한 **복합 안정화 지수**를 신규 제안하였습니다.

```
StabilizationIndex = α × (1 − dose_change_rate) + β × (1 / vital_cv)
```

- `dose_change_rate` : 약물 투여의 변동성을 나타내며, 값이 클수록 치료 과정이 불안정함을 의미
- `vital_cv` : 최근 48시간 생체 신호의 변동계수
- α, β 는 Grid Search로 최적값을 탐색하고, MAE Gradient Map으로 시각화하였습니다.

Ablation study 결과, Stabilization Index를 포함한 경우 baseline 대비 **MSE 0.0182 감소**를 달성하였습니다.

<br>

## 🤖 Modeling

Tree-based, Deep Learning, AutoML 모델을 비교 실험하였으며, Optuna를 활용하여 하이퍼파라미터 튜닝을 수행하였습니다.

| Model | Base MAE | Base RMSE | Tuned MAE | Tuned RMSE |
|---|---|---|---|---|
| Random Forest | 1.9844 | 3.1335 | 1.9744 | 3.1262 |
| LightGBM | 1.9399 | 3.0957 | 1.9347 | **3.0839** |
| XGBoost | 1.9668 | 3.1056 | **1.9331** | 3.0935 |
| CatBoost | 1.9732 | 3.1275 | 1.9373 | 3.1161 |
| TabNet | 2.1100 | 3.2817 | 1.9888 | 3.2737 |
| FT-Transformer | **1.870** | 3.0512 | **1.855** | 3.0408 |
| TabPFN | 1.8750 | 3.052 | **1.855** | 3.007 |
| **AutoGluon** | — | — | **1.5644** | **2.5657** |

AutoGluon은 LightGBM, CatBoost, XGBoost, NeuralNetTorch 계열 모델을 8-fold stacking ensemble로 결합하여 가장 우수한 성능을 달성하였습니다.

**Model Explainability**

SHAP 및 LIME을 활용하여 모델의 예측 근거를 분석하였습니다. GCS(의식 수준), medication_escalation, total_sedation_exposure, vasoactive_load 등이 LOS 예측에 주요한 영향을 미치는 것으로 확인되었습니다.

<br>

## 📊 Results

### Comparison with Reference Study (LOS ≤ 4 days)

기존 연구(Hempel, 2023)와의 비교입니다. 기존 연구는 입실 후 24시간 데이터를 사용하였고, 본 연구는 48시간 데이터를 사용하였으며 약물 투여군으로 코호트를 한정하였습니다.

| Method | RMSE | MAE |
|---|---|---|
| Linear Regression (Reference) | 1.07 | 0.74 |
| SVM (Reference) | 1.10 | 0.71 |
| Random Forest (Reference) | 1.07 | 0.74 |
| XGBoost (Reference) | 1.17 | 0.81 |
| **XGBoost (This study, ≤4 days)** | **0.81** | **0.65** |
| **AutoGluon (This study, ≤4 days)** | **0.74** | **0.58** |

### Binary Classification

| Method | Accuracy | F1 | AUC |
|---|---|---|---|
| XGBoost | 0.7750 | 0.5627 | 0.7912 |
| **AutoGluon** | **0.8266** | **0.6680** | **0.9224** |

<br>

## 🗃 Project Structure

```
MIMIC-IV-ICU-LoS/
│
├── code/
│   ├── eda_drug.ipynb
│   ├── eda_time.ipynb
│   │
│   ├── feature_engineering_vital.ipynb
│   ├── feature_engineering_lab.ipynb
│   ├── feature_engineering_drug.ipynb
│   ├── feature_engineering_drug_stabilityindex.ipynb
│   ├── feature_engineering_feature_select.ipynb
│   │
│   ├── modeling_rf_and_lgbm.ipynb
│   ├── modeling_Catboost_TabNet.ipynb
│   ├── modeling_TabPFN_and_FT_Transformer.ipynb
│   └── modeling_xgboost_final_XAI.ipynb
│
├── ReportandFAQ.pdf
├── requirements.txt
└── README.md
```

<br>

## 🛠 Tech Stack

**Core** : Python · pandas · numpy · scikit-learn

**Machine Learning** : XGBoost · LightGBM · CatBoost · Random Forest

**Deep Learning** : PyTorch · TabNet · FT-Transformer · TabPFN

**AutoML** : AutoGluon

**Hyperparameter Optimization** : Optuna

**Explainability** : SHAP · LIME

**Visualization** : matplotlib · seaborn