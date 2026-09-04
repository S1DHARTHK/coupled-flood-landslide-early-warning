# Kerala Flood–Landslide Early Warning

## Overview

Kerala is highly vulnerable to flood and landslide hazards, particularly during periods of prolonged and intense rainfall. These hazards can occur independently but may also develop under related environmental conditions.

This project presents a **lead-time-oriented early warning framework for coupled flood–landslide hazards in Kerala**. The system uses two independently trained XGBoost models: one for flood prediction and another for landslide prediction. Their outputs are then passed to a **rule-based Temporal Coupled Decision Layer (TCDL)**, which analyses the temporal trends of both hazards along with rainfall and soil-moisture trends to generate an early-warning decision.

The main objective is to investigate whether **temporally coupling independently predicted flood and landslide risks can provide earlier warnings than treating the hazards independently**. Therefore, warning **lead time is the primary evaluation measure**, supported by standard classification metrics such as precision, recall, F1-score, ROC-AUC, and PR-AUC.

> **Current Status:** The current implementation uses synthetic data for development and testing of the complete pipeline. Real-world datasets and historical event timelines will be used for final evaluation.



## System Architecture


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


The framework consists of two independent hazard-specific machine learning models followed by a rule-based temporal coupling layer.

The **Flood XGBoost model** predicts the probability of flooding using common environmental features and flood-specific hydrological features.

The **Landslide XGBoost model** predicts the probability of landsliding using common environmental features and landslide-specific terrain and geological features.

The outputs of both models are provided to the **Temporal Coupled Decision Layer (TCDL)**. TCDL analyses the probability trends of both hazards together with rainfall and soil-moisture trends and applies predefined temporal coupling rules to generate the final warning decision.

TCDL is **not a third machine learning model**. It is a transparent, rule-based decision layer designed to perform temporal and cross-hazard fusion of the outputs from the two independent models.



## Key Components

### 1. Flood XGBoost Model

The flood model is an independent binary classification model that estimates the probability of a flood occurring at a given location and time.

It uses common environmental features together with flood-specific hydrological features such as river level, distance to river, and drainage density.

**Output:**

* Flood probability
* Flood probability trend

### 2. Landslide XGBoost Model

The landslide model is an independent binary classification model that estimates the probability of a landslide occurring at a given location and time.

It uses common environmental features together with landslide-specific terrain and geological features such as slope, aspect, curvature, land cover, and lithology.

**Output:**

* Landslide probability
* Landslide probability trend

### 3. Temporal Coupled Decision Layer (TCDL)

TCDL is the coupling component of the framework. It receives the outputs of the flood and landslide models and evaluates how the risks change over time.

The layer considers:

* Flood probability trend
* Landslide probability trend
* Rainfall trend
* Soil-moisture trend
* Combined flood–landslide risk

Based on these temporal and cross-hazard conditions, TCDL generates warning decisions such as:

* No Warning
* Flood Warning
* Landslide Warning
* Coupled Hazard Warning

The purpose of TCDL is to investigate whether combining the evolving risk signals of both hazards can provide useful warning time before a historical disaster event.

### 4. Lead-Time Evaluation

Lead time is the primary evaluation focus of the project.

The framework measures the time between the generated warning and the occurrence of the corresponding historical event:


Lead Time = Actual Event Time − Warning Time


The coupled TCDL approach is evaluated against the independent flood-only and landslide-only approaches to determine whether temporal coupling can provide earlier warnings.

### 5. Historical Event Evaluation

Historical flood and landslide events are used as ground truth for retrospective evaluation. The system's generated warnings can be compared with documented event times and, where available, historical warning timelines.

This allows the project to evaluate the operational usefulness of the framework in terms of warning lead time rather than relying only on conventional classification performance.


This is a good stopping point for now. **Don't add the detailed TCDL rules here yet**—those are better placed later under a dedicated `TCDL Methodology` section.

