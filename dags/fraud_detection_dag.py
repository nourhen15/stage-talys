from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'nourhen',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fraud_detection_pipeline',
    default_args=default_args,
    description='Pipeline ML pour la détection de fraude bancaire',
    schedule_interval=None,  # Déclenchement manuel uniquement (pas de run automatique quotidien)
    catchup=False
)

# Chemins partagés (correspondent au volume ./data:/opt/airflow/data)
RAW_PATH = "/opt/airflow/data/raw/initial_data.csv"
PROCESSED_PATH = "/opt/airflow/data/processed/processed_data.csv"
MLFLOW_TRACKING_DIR = "/opt/airflow/data/mlruns"
MODEL_EXPORT_PATH = "/opt/airflow/data/model/model.pkl"


def extract_data(**kwargs):
    import pandas as pd

    df = pd.read_csv(RAW_PATH)
    print(f"Données extraites : {len(df)} lignes")

    # On pousse juste le chemin du fichier via XCom, pas les données elles-mêmes
    kwargs['ti'].xcom_push(key='raw_path', value=RAW_PATH)


def preprocess_data(**kwargs):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import os

    raw_path = kwargs['ti'].xcom_pull(key='raw_path', task_ids='extract_data')
    df = pd.read_csv(raw_path)

    # 'Amount' et 'Time' ne sont pas issues de la PCA -> on les met à la même échelle
    # que les autres colonnes (V1...V28), sinon le modèle leur donnerait trop d'importance
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time'] = scaler.fit_transform(df[['Time']])

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Données prétraitées sauvegardées : {PROCESSED_PATH}")

    kwargs['ti'].xcom_push(key='processed_path', value=PROCESSED_PATH)


def train_model(**kwargs):
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    import mlflow
    import mlflow.sklearn

    processed_path = kwargs['ti'].xcom_pull(key='processed_path', task_ids='preprocess_data')
    df = pd.read_csv(processed_path)

    X = df.drop(columns=['Class'])
    y = df['Class']

    # stratify=y : on garde la même proportion de fraudes dans train et test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment("fraud_detection")

    # --- Étape 1 : comparer 5 modèles sur un échantillon réduit ---
    # (comparer sur les 227k lignes complètes avec cross-validation serait beaucoup
    # trop long pour certains modèles ; un échantillon stratifié de 50 000 lignes
    # donne une comparaison fiable en une fraction du temps)
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train, train_size=50000, random_state=42, stratify=y_train
    )

    # scale_pos_weight : équivalent pour XGBoost du class_weight='balanced' des autres
    # modèles (XGBoost n'a pas ce paramètre directement) -> ratio non-fraude / fraude
    ratio_desequilibre = (y_sample == 0).sum() / (y_sample == 1).sum()

    candidats = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "decision_tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight='balanced', random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "xgboost": XGBClassifier(
            scale_pos_weight=ratio_desequilibre, eval_metric='logloss', random_state=42
        ),
    }

    with mlflow.start_run(run_name="model_selection") as parent_run:
        scores = {}

        for nom, modele in candidats.items():
            with mlflow.start_run(run_name=nom, nested=True):
                # cv=3 : validation croisée à 3 plis, scoring='f1' car c'est la métrique
                # la plus parlante ici (le dataset est très déséquilibré)
                cv_scores = cross_val_score(modele, X_sample, y_sample, cv=3, scoring='f1')
                score_moyen = cv_scores.mean()
                scores[nom] = score_moyen

                mlflow.log_param("modele", nom)
                mlflow.log_metric("f1_cv_moyen", score_moyen)
                print(f"{nom} : F1 moyen (cross-validation) = {score_moyen:.3f}")

        # --- Étape 2 : choisir le meilleur et le réentraîner sur TOUTES les données ---
        meilleur_nom = max(scores, key=scores.get)
        print(f"\nMeilleur modèle : {meilleur_nom} (F1 cv = {scores[meilleur_nom]:.3f})")

        meilleur_modele = candidats[meilleur_nom]
        meilleur_modele.fit(X_train, y_train)

        mlflow.log_param("meilleur_modele", meilleur_nom)
        mlflow.log_metric("f1_cv_meilleur", scores[meilleur_nom])
        mlflow.sklearn.log_model(meilleur_modele, "model")

        run_id = parent_run.info.run_id
        print(f"Modèle final entraîné et loggé. Run MLflow : {run_id}")

        # On sauvegarde aussi le jeu de test pour que evaluate_model puisse s'en servir
        X_test.to_csv("/opt/airflow/data/processed/X_test.csv", index=False)
        y_test.to_csv("/opt/airflow/data/processed/y_test.csv", index=False)

    kwargs['ti'].xcom_push(key='run_id', value=run_id)


def evaluate_model(**kwargs):
    import pandas as pd
    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
    import mlflow
    import mlflow.sklearn

    run_id = kwargs['ti'].xcom_pull(key='run_id', task_ids='train_model')

    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    X_test = pd.read_csv("/opt/airflow/data/processed/X_test.csv")
    y_test = pd.read_csv("/opt/airflow/data/processed/y_test.csv").squeeze()

    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

    # On rattache les métriques d'évaluation au même run MLflow
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)


def export_model(**kwargs):
    import joblib
    import mlflow
    import mlflow.sklearn
    import os
    import json

    run_id = kwargs['ti'].xcom_pull(key='run_id', task_ids='train_model')

    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    os.makedirs(os.path.dirname(MODEL_EXPORT_PATH), exist_ok=True)

    # joblib est le format standard pour sauvegarder des modèles scikit-learn de façon
    # légère et rapide à recharger -> l'API n'aura besoin que de ce seul fichier,
    # sans dépendre de MLflow au moment de servir les prédictions
    joblib.dump(model, MODEL_EXPORT_PATH)

    # On sauvegarde aussi la liste des colonnes attendues (dans le bon ordre) : l'API
    # en aura besoin pour valider et ordonner correctement les données reçues
    colonnes = list(model.feature_names_in_)
    with open("/opt/airflow/data/model/feature_columns.json", "w") as f:
        json.dump(colonnes, f)

    print(f"Modèle exporté vers {MODEL_EXPORT_PATH}")
    print(f"Run MLflow source : {run_id}")


extract = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

export = PythonOperator(
    task_id='export_model',
    python_callable=export_model,
    dag=dag
)

extract >> preprocess >> train >> evaluate >> export