# Fraud Detection MLOps Platform

## 1. Présentation

Ce projet consiste à mettre en place une plateforme MLOps complète pour la détection automatique des transactions frauduleuses.

L'objectif est d'automatiser l'ensemble du cycle de vie du modèle :

EDA → Preprocessing → Training → Evaluation → Validation → Export → API → Monitoring → Drift Detection → Retraining

Le projet utilise Docker pour l'environnement, Apache Airflow pour l'orchestration, MLflow pour le suivi des expériences, FastAPI pour le déploiement du modèle, Prometheus/Grafana pour le monitoring et DVC pour la gestion des données et des modèles.

---

## 2. Architecture

```text
                         ┌──────────────────────┐
                         │       Dataset        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │         EDA          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Preprocessing     │
                         │  + Feature Engineering│
                         │  + CTGAN              │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Training       │
                         │ Model Comparison     │
                         │ Cross Validation     │
                         │ Hyperparameter Tuning│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Evaluation      │
                         │ F1 / Recall / AUC    │
                         │ Precision / Accuracy │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Validation Gate    │
                         └──────────┬───────────┘
                                    │
                           ┌────────┴────────┐
                           │                 │
                         Reject            Accept
                                             │
                                             ▼
                                  ┌──────────────────┐
                                  │   Export Model   │
                                  └────────┬─────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │     FastAPI      │
                                  │     /predict     │
                                  └────────┬─────────┘
                                           │
                              ┌────────────┴────────────┐
                              │                         │
                              ▼                         ▼
                         Prometheus                 Grafana
                              │
                              ▼
                         Monitoring

                    New Data
                       │
                       ▼
                Drift Monitoring
                       │
                 ┌─────┴─────┐
                 │           │
              No Drift      Drift
                 │           │
                Stop         ▼
                       Retraining
                            │
                            ▼
                    Fraud Detection DAG