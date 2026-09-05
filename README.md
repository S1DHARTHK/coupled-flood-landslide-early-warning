
# Kerala Flood–Landslide Early Warning

## 1. Overview

Kerala is highly vulnerable to flood and landslide hazards, particularly during periods of prolonged and intense rainfall. These hazards can occur independently but may also develop under related environmental conditions.

This project presents a **lead-time-oriented early warning framework for coupled flood–landslide hazards in Kerala**. The system uses two independently trained XGBoost models: one for flood prediction and another for landslide prediction. Their outputs are then passed to a **rule-based Temporal Coupled Decision Layer (TCDL)**, which analyses the temporal trends of both hazards along with rainfall and soil-moisture trends to generate an early-warning decision.

The main objective is to investigate whether **temporally coupling independently predicted flood and landslide risks can provide earlier warnings than treating the hazards independently**. Warning **lead time is the primary evaluation focus**, supported by standard classification metrics such as precision, recall, F1-score, ROC-AUC, and PR-AUC.

> **Current Status:** The current implementation uses synthetic data for development and testing of the complete pipeline. Real-world datasets and historical event timelines will be used for final evaluation.

---

## 2. System Architecture

```text
                    Environmental Data
                           │
                           ▼
                    Master Dataset
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Flood XGBoost             Landslide XGBoost
              │                         │
              ▼                         ▼
       Flood Probability        Landslide Probability
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
             Temporal Coupled Decision
                    Layer (TCDL)
                           │
                           ▼
                  Warning Decision
                           │
                           ▼
                 Lead-Time Evaluation
````

The framework consists of **two independent hazard-specific machine learning models followed by a rule-based temporal coupling layer**.

The Flood XGBoost model predicts flood probability using common environmental features and flood-specific hydrological features.

The Landslide XGBoost model predicts landslide probability using common environmental features and landslide-specific terrain and geological features.

The outputs from both models are passed to the **Temporal Coupled Decision Layer (TCDL)**. TCDL analyses the changing probability trends of both hazards together with rainfall and soil-moisture trends to generate the final warning decision.

---

## 3. Key Components

### 3.1 Flood XGBoost Model

The Flood XGBoost model is an independent binary classification model that predicts the probability of flooding at a given location and time.

It uses common environmental features together with flood-specific features such as:

* River level
* Distance to river
* Drainage density
* Antecedent rainfall
* Soil moisture

**Output:**

* Flood probability
* Flood probability trend

### 3.2 Landslide XGBoost Model

The Landslide XGBoost model is an independent binary classification model that predicts the probability of landsliding at a given location and time.

It uses common environmental features together with landslide-specific terrain and geological features such as:

* Slope
* Aspect
* Curvature
* Land cover
* Lithology
* Antecedent rainfall
* Soil moisture

**Output:**

* Landslide probability
* Landslide probability trend

### 3.3 Temporal Coupled Decision Layer (TCDL)

The **Temporal Coupled Decision Layer (TCDL)** is the coupling component of the framework.

TCDL is **not a third machine learning model**. It is a rule-based temporal decision layer that combines the outputs of the two independent XGBoost models with environmental trends.

It considers:

* Flood probability trend
* Landslide probability trend
* Rainfall trend
* Soil-moisture trend
* Combined flood–landslide risk

Based on these temporal and cross-hazard conditions, TCDL generates the final warning decision, such as:

* No Warning
* Flood Warning
* Landslide Warning
* Coupled Hazard Warning

The purpose of TCDL is to investigate whether combining the evolving risk signals of both hazards can provide useful warning time before a historical disaster event.


