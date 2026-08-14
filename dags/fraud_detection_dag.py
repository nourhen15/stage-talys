from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
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
    # Déclenchement manuel uniquement : un schedule automatique (@daily) ajouterait
    # une charge régulière que la machine (8 Go RAM) ne supporte pas confortablement
    schedule_interval=None,
    catchup=False
)

# Chemins partagés (correspondent au volume ./data:/opt/airflow/data)
RAW_PATH = "/opt/airflow/data/raw/initial_data.csv"
NEW_DATA_PATH = "/opt/airflow/data/raw/new_data.csv"
PROCESSED_PATH = "/opt/airflow/data/processed/processed_data.csv"
MLFLOW_TRACKING_DIR = "/opt/airflow/data/mlruns"
MODEL_EXPORT_PATH = "/opt/airflow/data/model/model.pkl"
SCALER_EXPORT_PATH = "/opt/airflow/data/model/preprocessing_scaler.pkl"


def extract_data(**kwargs):
    import pandas as pd

    # Permet de distinguer un run manuel d'un retraining déclenché par le
    # DAG de monitoring lorsqu'une dérive est détectée
    dag_run = kwargs.get("dag_run")

    use_new_data = (
        dag_run
        and dag_run.conf
        and dag_run.conf.get("use_new_data", False)
    )

    if use_new_data:
        df_initial = pd.read_csv(RAW_PATH)
        df_new = pd.read_csv(NEW_DATA_PATH)

        # On combine les données historiques avec les nouvelles données
        # pour effectuer un véritable retraining
        df = pd.concat(
            [df_initial, df_new],
            ignore_index=True
        )

        retraining_path = "/opt/airflow/data/raw/retraining_data.csv"

        df.to_csv(
            retraining_path,
            index=False
        )

        print(
            f"Retraining avec initial_data + new_data : "
            f"{len(df)} lignes"
        )

        kwargs['ti'].xcom_push(
            key='raw_path',
            value=retraining_path
        )

    else:
        df = pd.read_csv(RAW_PATH)
        print(f"Données extraites : {len(df)} lignes")

        kwargs['ti'].xcom_push(
            key='raw_path',
            value=RAW_PATH
        )


def preprocess_data(**kwargs):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import os

    raw_path = kwargs['ti'].xcom_pull(key='raw_path', task_ids='extract_data')
    df = pd.read_csv(raw_path)

    # Découverte de l'EDA : 718 lignes dupliquées -> on les retire
    n_avant = len(df)
    df = df.drop_duplicates()
    print(f"Doublons retirés : {n_avant - len(df)} lignes")

    # Découverte de l'EDA : le taux de fraude varie fortement selon l'heure
    df['Hour'] = (df['Time'] // 3600) % 24

    # On sauvegarde les statistiques de référence AVANT le scaling.
    # Cela permet de comparer les futures données brutes avec les données
    # de référence brutes dans le DAG de surveillance de la dérive.
    import json

    # On utilise initial_data comme référence stable pour le monitoring,
    # même lorsqu'un retraining est déclenché avec new_data.
    df_reference = pd.read_csv(RAW_PATH)
    df_reference = df_reference.drop_duplicates()

    features_surveillees = ['V14', 'V4', 'V12', 'V10', 'V17', 'Amount']

    stats_reference = {
        col: {
            "moyenne": float(df_reference[col].mean()),
            "ecart_type": float(df_reference[col].std())
        }
        for col in features_surveillees
    }

    os.makedirs("/opt/airflow/data/monitoring", exist_ok=True)

    with open("/opt/airflow/data/monitoring/reference_stats.json", "w") as f:
        json.dump(stats_reference, f, indent=2)

    print("Statistiques de référence sauvegardées pour la détection de dérive")

    # Le scaler est entraîné sur les données utilisées pour ce run.
    # Il sera sauvegardé puis réutilisé par FastAPI afin d'appliquer
    # exactement la même transformation aux données reçues en production.
    scaler = StandardScaler()

    df[['Amount', 'Time']] = scaler.fit_transform(
        df[['Amount', 'Time']]
    )

    os.makedirs("/opt/airflow/data/model", exist_ok=True)

    import joblib

    joblib.dump(
        scaler,
        SCALER_EXPORT_PATH
    )

    print(
        f"Scaler sauvegardé : {SCALER_EXPORT_PATH}"
    )

    # --- Génération de fraudes synthétiques avec CTGAN (demande de l'encadrant) ---
    # On entraîne le GAN UNIQUEMENT sur les transactions frauduleuses (peu nombreuses,
    # ~400 lignes) -> beaucoup plus léger que d'entraîner sur tout le dataset, puisque
    # le but est justement d'apprendre à quoi ressemble UNE fraude pour en générer
    # d'autres, pas d'apprendre tout le dataset.
    from ctgan import CTGAN

    colonnes_numeriques = [c for c in df.columns if c not in ('Class',)]
    df_fraudes = df[df['Class'] == 1][colonnes_numeriques]
    print(f"Entraînement du CTGAN sur {len(df_fraudes)} transactions frauduleuses...")

    # epochs volontairement bas (défaut CTGAN = 300) : suffisant pour un premier
    # résultat exploitable, tout en restant raisonnable en temps/mémoire sur une
    # machine aux ressources limitées
    gan = CTGAN(epochs=100, batch_size=100, verbose=False)
    gan.fit(df_fraudes)

    # On génère autant de fraudes synthétiques que de vraies fraudes -> double le
    # nombre d'exemples de fraude disponibles pour l'entraînement, sans pour autant
    # sur-représenter artificiellement la classe minoritaire
    n_synthetiques = len(df_fraudes)
    fraudes_synthetiques = gan.sample(n_synthetiques)
    fraudes_synthetiques['Class'] = 1
    print(f"{n_synthetiques} fraudes synthétiques générées")

    df = pd.concat([df, fraudes_synthetiques], ignore_index=True)
    print(f"Dataset après ajout des fraudes synthétiques : {len(df)} lignes "
          f"({df['Class'].sum()} fraudes au total)")

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Données prétraitées sauvegardées : {PROCESSED_PATH}")

    kwargs['ti'].xcom_push(key='processed_path', value=PROCESSED_PATH)


def train_model(**kwargs):
    import gc
    import time
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    import mlflow
    import mlflow.sklearn

    processed_path = kwargs['ti'].xcom_pull(key='processed_path', task_ids='preprocess_data')

    # float32 au lieu du float64 par défaut : divise par 2 la mémoire utilisée pour
    # charger les données (important ici, la machine a des ressources limitées, 8 Go RAM)
    colonnes_v = [f"V{i}" for i in range(1, 29)]
    dtypes = {col: 'float32' for col in colonnes_v + ['Time', 'Amount', 'Hour']}
    df = pd.read_csv(processed_path, dtype=dtypes)

    X = df.drop(columns=['Class'])
    y = df['Class']
    del df
    gc.collect()

    # stratify=y : on garde la même proportion de fraudes dans train et test
    X_train_complet, X_test, y_train_complet, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    del X, y
    gc.collect()

    # --- Contrainte "modèle efficient" (demande de l'encadrant) : on entraîne le
    # modèle final sur un sous-échantillon de 80 000 lignes plutôt que sur les 182 000
    # lignes complètes. Vu la machine disponible (8 Go RAM) et le fait que le dataset
    # est déjà très séparable (V14 notamment, cf. EDA), ce compromis reste largement
    # suffisant pour de bonnes performances, tout en réduisant nettement le temps et
    # la mémoire nécessaires -> c'est un choix d'efficience assumé, pas une limitation
    # cachée.
    X_train, _, y_train, _ = train_test_split(
        X_train_complet, y_train_complet, train_size=80000, random_state=42, stratify=y_train_complet
    )
    del X_train_complet, y_train_complet
    gc.collect()

    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment("fraud_detection")

    # --- Étape 1 : comparer 5 modèles baseline sur un échantillon réduit ---
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train, train_size=30000, random_state=42, stratify=y_train
    )

    # scale_pos_weight : équivalent pour XGBoost du class_weight='balanced'
    ratio_desequilibre = (y_sample == 0).sum() / (y_sample == 1).sum()

    candidats = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "decision_tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=50, max_depth=10, class_weight='balanced', random_state=42, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=50, scale_pos_weight=ratio_desequilibre, eval_metric='logloss',
            random_state=42, n_jobs=1, tree_method='hist'
        ),
    }

    # --- Grilles d'hyperparamètres, volontairement compactes ---
    grilles = {
        "logistic_regression": {"C": [0.01, 0.1, 1, 10]},
        "decision_tree": {"max_depth": [5, 10, 15], "min_samples_leaf": [1, 5, 10]},
        "random_forest": {"n_estimators": [50, 100], "max_depth": [5, 10]},
        "gradient_boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
        "xgboost": {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    }

    with mlflow.start_run(run_name="model_selection") as parent_run:
        scores = {}

        for nom, modele in candidats.items():
            with mlflow.start_run(run_name=nom, nested=True):
                cv_scores = cross_val_score(modele, X_sample, y_sample, cv=3, scoring='f1', n_jobs=1)
                score_moyen = cv_scores.mean()
                scores[nom] = score_moyen

                mlflow.log_param("modele", nom)
                mlflow.log_metric("f1_cv_moyen", score_moyen)
                print(f"{nom} : F1 moyen (cross-validation) = {score_moyen:.3f}")

        del X_sample, y_sample
        gc.collect()

        meilleur_nom = max(scores, key=scores.get)
        print(f"\nMeilleur modèle (baseline) : {meilleur_nom} (F1 cv = {scores[meilleur_nom]:.3f})")

        # --- RandomizedSearchCV léger : n_iter=3 combinaisons x cv=3 = 9 entraînements
        # seulement, sur l'échantillon de 80 000 lignes ---
        print(f"Recherche d'hyperparamètres pour {meilleur_nom}...")
        grille = grilles[meilleur_nom]
        modele_de_base = candidats[meilleur_nom]

        recherche = RandomizedSearchCV(
            modele_de_base, grille, n_iter=3, cv=3, scoring='f1',
            n_jobs=1, random_state=42
        )
        recherche.fit(X_train, y_train)

        print(f"Meilleurs hyperparamètres trouvés : {recherche.best_params_}")
        print(f"Meilleur F1 (recherche) : {recherche.best_score_:.3f}")

        meilleur_modele = recherche.best_estimator_
        del recherche
        gc.collect()

        mlflow.log_param("meilleur_modele", meilleur_nom)
        mlflow.log_param("taille_echantillon_entrainement", len(X_train))
        mlflow.log_metric("f1_cv_baseline", scores[meilleur_nom])

        # --- Efficience : taille du modèle et latence de prédiction ---
        import joblib
        import tempfile
        import os as _os
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            joblib.dump(meilleur_modele, tmp.name)
            taille_ko = _os.path.getsize(tmp.name) / 1024
            _os.remove(tmp.name)
        mlflow.log_metric("taille_modele_ko", taille_ko)

        debut = time.time()
        meilleur_modele.predict(X_test.iloc[:1000])
        latence_s = time.time() - debut
        mlflow.log_metric("latence_moyenne_ms", latence_s)
        print(f"Taille du modèle : {taille_ko:.1f} Ko | Latence (1000 préd.) : {latence_s*1000:.1f} ms")

        mlflow.sklearn.log_model(meilleur_modele, "model")

        run_id = parent_run.info.run_id
        print(f"Modèle final entraîné et loggé. Run MLflow : {run_id}")

        X_test.to_csv("/opt/airflow/data/processed/X_test.csv", index=False)
        y_test.to_csv("/opt/airflow/data/processed/y_test.csv", index=False)

    kwargs['ti'].xcom_push(key='run_id', value=run_id)


def evaluate_model(**kwargs):
    import pandas as pd
    from sklearn.metrics import (
        classification_report, f1_score, precision_score, recall_score,
        accuracy_score, roc_auc_score
    )
    import mlflow
    import mlflow.sklearn

    run_id = kwargs['ti'].xcom_pull(key='run_id', task_ids='train_model')

    mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    X_test = pd.read_csv("/opt/airflow/data/processed/X_test.csv")
    y_test = pd.read_csv("/opt/airflow/data/processed/y_test.csv").squeeze()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # AUC-ROC : mesure la capacité du modèle à séparer les 2 classes, indépendamment
    # d'un seuil de décision précis -> aide à diagnostiquer under/overfitting
    auc_roc = roc_auc_score(y_test, y_proba)

    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | "
          f"F1: {f1:.3f} | AUC-ROC: {auc_roc:.3f}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc_roc)

    # Les métriques sont envoyées via XCom afin que la tâche de validation
    # puisse décider si le modèle peut être exporté.
    kwargs['ti'].xcom_push(
        key='f1_score',
        value=float(f1)
    )

    kwargs['ti'].xcom_push(
        key='recall',
        value=float(recall)
    )


def validate_model(**kwargs):
    ti = kwargs['ti']

    f1 = ti.xcom_pull(
        key='f1_score',
        task_ids='evaluate_model'
    )

    recall = ti.xcom_pull(
        key='recall',
        task_ids='evaluate_model'
    )

    # Seuils minimums de validation du modèle
    MIN_F1 = 0.75
    MIN_RECALL = 0.75

    print(
        f"Validation du modèle : "
        f"F1={f1:.3f} | Recall={recall:.3f}"
    )

    if f1 >= MIN_F1 and recall >= MIN_RECALL:
        print("✅ Modèle validé")
        return "export_model"

    print("❌ Modèle rejeté : performances insuffisantes")
    return "reject_model"


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
    joblib.dump(model, MODEL_EXPORT_PATH)

    # On vérifie que le scaler généré pendant le preprocessing existe avant
    # d'autoriser l'export du modèle.
    if not os.path.exists(SCALER_EXPORT_PATH):
        raise FileNotFoundError(
            f"Scaler introuvable : {SCALER_EXPORT_PATH}"
        )

    colonnes = list(model.feature_names_in_)
    with open("/opt/airflow/data/model/feature_columns.json", "w") as f:
        json.dump(colonnes, f)

    print(f"Modèle exporté vers {MODEL_EXPORT_PATH}")
    print(f"Scaler exporté vers {SCALER_EXPORT_PATH}")
    print(f"Run MLflow source : {run_id}")


extract = PythonOperator(task_id='extract_data', python_callable=extract_data, dag=dag)
preprocess = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data, dag=dag)
train = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)
evaluate = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, dag=dag)

validate = BranchPythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

export = PythonOperator(task_id='export_model', python_callable=export_model, dag=dag)

reject = EmptyOperator(
    task_id='reject_model',
    dag=dag
)


extract >> preprocess >> train >> evaluate >> validate

validate >> export
validate >> reject