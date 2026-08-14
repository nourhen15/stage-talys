"""
DAG de surveillance de la dérive des données (data drift).

Compare les statistiques des nouvelles données (new_data.csv, qui simule des
transactions arrivées après la période d'entraînement initiale) à celles utilisées
pour entraîner le modèle actuel. Si l'écart dépasse un seuil, ça veut dire que les
patterns de fraude ont probablement changé -> le DAG principal est automatiquement
déclenché pour réentraîner le modèle sur des données à jour.

Ce DAG est volontairement léger (juste des calculs de statistiques, pas de calcul
lourd) pour tourner fréquemment sans consommer trop de ressources.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'nourhen',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'drift_monitoring_pipeline',
    default_args=default_args,
    description='Surveillance de la dérive des données, déclenche le réentraînement si nécessaire',
    # Toutes les 6h : suffisant pour détecter une dérive rapidement, sans surcharger
    # une machine aux ressources limitées avec des vérifications trop fréquentes
    schedule_interval=timedelta(hours=6),
    catchup=False
)

NEW_DATA_PATH = "/opt/airflow/data/raw/new_data.csv"
REFERENCE_STATS_PATH = "/opt/airflow/data/monitoring/reference_stats.json"

# Seuil de dérive : au-delà de 2 écarts-types de différence entre la moyenne de
# référence et celle des nouvelles données, on considère qu'il y a une dérive
# significative (règle statistique courante, ajustable selon les besoins réels)
SEUIL_DERIVE = 2.0


def detecter_derive(**kwargs):
    import pandas as pd
    import json

    # Charge les statistiques de référence, calculées lors du dernier entraînement
    with open(REFERENCE_STATS_PATH) as f:
        stats_reference = json.load(f)

    # Charge les "nouvelles" données (simulation : new_data.csv joue le rôle de
    # données fraîchement arrivées, jamais vues par le modèle actuel)
    df_nouveau = pd.read_csv(NEW_DATA_PATH)

    derive_detectee = False
    details = {}

    for feature, ref in stats_reference.items():
        moyenne_nouvelle = df_nouveau[feature].mean()
        # Écart normalisé : de combien d'écarts-types de référence la nouvelle
        # moyenne s'est décalée -> une mesure simple mais efficace de dérive
        ecart_normalise = abs(moyenne_nouvelle - ref["moyenne"]) / (ref["ecart_type"] + 1e-9)
        details[feature] = round(ecart_normalise, 3)

        if ecart_normalise > SEUIL_DERIVE:
            derive_detectee = True

    print("Écarts normalisés par feature :", details)
    print(f"Dérive détectée : {derive_detectee} (seuil = {SEUIL_DERIVE})")

    # Pousse le résultat pour la tâche suivante (qui décide s'il faut déclencher
    # le réentraînement)
    kwargs['ti'].xcom_push(key='derive_detectee', value=derive_detectee)
    kwargs['ti'].xcom_push(key='details_derive', value=details)


def decider_declenchement(**kwargs):
    """Décide s'il faut déclencher le réentraînement, en fonction de la dérive détectée."""
    ti = kwargs['ti']
    derive_detectee = ti.xcom_pull(key='derive_detectee', task_ids='detecter_derive')

    if derive_detectee:
        return 'declencher_reentrainement'
    return 'pas_de_derive'


detection = PythonOperator(
    task_id='detecter_derive',
    python_callable=detecter_derive,
    dag=dag
)

from airflow.operators.python import BranchPythonOperator

decision = BranchPythonOperator(
    task_id='decider_declenchement',
    python_callable=decider_declenchement,
    dag=dag
)

declenchement = TriggerDagRunOperator(
    task_id='declencher_reentrainement',
    trigger_dag_id='fraud_detection_pipeline',
    conf={
        'use_new_data': True
    },
    dag=dag
)

from airflow.operators.empty import EmptyOperator

pas_de_derive = EmptyOperator(
    task_id='pas_de_derive',
    dag=dag
)

detection >> decision >> [declenchement, pas_de_derive]