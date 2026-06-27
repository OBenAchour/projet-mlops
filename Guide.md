# Guide du projet — Description étape par étape

Ce document décrit chaque étape du projet MLOps Customer Churn, avec une présentation des technologies utilisées.

---

## Technologies utilisées

**Python** — Langage de programmation principal du projet. Utilisé pour le traitement de données, le machine learning, l'API et l'orchestration. Écosystème riche en librairies scientifiques et ML.

**scikit-learn** — Bibliothèque Python de machine learning. Fournit des algorithmes prêts à l'emploi (RandomForest, SVM, etc.), des outils de preprocessing (StandardScaler, LabelEncoder) et des métriques d'évaluation (accuracy, precision, recall, F1).

**FastAPI** — Framework web Python moderne et performant pour créer des API REST. Génère automatiquement une documentation Swagger interactive. Basé sur les type hints Python et la validation Pydantic.

**Uvicorn** — Serveur ASGI ultra-rapide pour exécuter les applications FastAPI. Supporte le rechargement automatique en développement.

**Prefect** — Plateforme d'orchestration de workflows Python. Permet de définir des tasks (fonctions unitaires) et des flows (enchaînements de tasks), de les scheduler, monitorer et relancer en cas d'échec. Fournit une UI web pour visualiser les exécutions.

**MLflow** — Plateforme open source pour le cycle de vie du ML. Permet de tracker les expériences (paramètres, métriques, artefacts), versionner les modèles, et comparer les runs via une interface web.

**Docker** — Plateforme de conteneurisation. Permet d'empaqueter l'application avec toutes ses dépendances dans une image portable, exécutable sur n'importe quelle machine. Docker Hub sert de registre pour partager les images.

**Docker Compose** — Outil pour définir et gérer des applications multi-conteneurs. Un fichier YAML décrit les services (Elasticsearch, Kibana), leurs configurations et leurs réseaux.

**Elasticsearch** — Moteur de recherche et d'analyse distribué. Stocke les données sous forme de documents JSON indexés. Utilisé ici pour centraliser les métriques et logs du pipeline ML.

**Kibana** — Interface de visualisation pour Elasticsearch. Permet d'explorer les données via Discover, créer des dashboards et des graphiques pour monitorer les performances du modèle.

**Black** — Formateur de code Python automatique. Applique un style cohérent sans configuration, éliminant les débats de formatage.

**Flake8** — Vérificateur de style Python. Détecte les violations PEP8 (lignes trop longues, imports inutilisés, etc.).

**Pylint** — Analyseur de qualité de code Python approfondi. Vérifie les conventions, les erreurs potentielles et la complexité du code.

**Bandit** — Scanner de sécurité pour Python. Détecte les vulnérabilités courantes (subprocess sans validation, mots de passe hardcodés, etc.).

**Pytest** — Framework de tests unitaires Python. Simple, extensible, avec des rapports détaillés et le support des fixtures.

**WSL (Windows Subsystem for Linux)** — Couche de compatibilité permettant d'exécuter un environnement Linux natif sous Windows. Utilisé pour développer dans un environnement proche de la production.

---

## Étape 1 : Mise en place de l'environnement

Le projet tourne sous WSL avec Ubuntu. On crée un environnement virtuel Python (venv) pour isoler les dépendances. Le fichier requirements.txt liste toutes les librairies nécessaires : pandas et numpy pour la manipulation de données, scikit-learn pour le ML, matplotlib pour les visualisations, et les outils de qualité de code.

---

## Étape 2 : Code modulaire

Le code est séparé en modules avec des responsabilités claires :

**data_loader.py** charge le dataset Churn_Modelling.csv (10 000 clients bancaires), encode la colonne Gender avec LabelEncoder, supprime les colonnes non pertinentes (Surname, Geography, RowNumber, CustomerId), fait un split 80/20 train/test, et normalise les features avec StandardScaler.

**trainer.py** entraîne un RandomForestClassifier (100 arbres de décision). Il log les hyperparamètres dans MLflow pour le suivi.

**evaluator.py** calcule 4 métriques sur le jeu de test : accuracy, precision, recall et F1-score. Il génère une matrice de confusion en PNG et log tout dans MLflow.

**model_io.py** sauvegarde le modèle au format joblib (sérialisation optimisée pour les objets NumPy/scikit-learn) et le recharge pour la prédiction.

**main.py** orchestre le tout dans un contexte MLflow, avec la possibilité de lancer des étapes individuelles via des arguments en ligne de commande.

---

## Étape 3 : Pipeline Prefect (Atelier 3)

Le fichier pipeline_prefect.py transforme chaque fonction en task Prefect (décorateur @task) et les regroupe en flows (décorateur @flow). Les flows disponibles sont : all (pipeline complet), install, code, test, prepare, train, evaluate, predict, clean et api.

Le fichier deploiement_prefect.py enregistre les flows comme des déploiements Prefect, permettant le scheduling automatique (le pipeline complet est planifié tous les jours à 2h du matin via un cron).

L'architecture nécessite 3 terminaux pour le déploiement : le serveur Prefect (UI port 4200), le worker qui exécute les flows, et un terminal pour déclencher manuellement les déploiements.

---

## Étape 4 : API FastAPI (Atelier 4)

Le fichier app.py expose le modèle comme un service REST. Au démarrage, il charge le modèle classifier.joblib en mémoire.

L'endpoint POST /predict reçoit les 9 features d'un client (CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary) et retourne la prédiction (churn ou pas) avec les probabilités.

L'endpoint POST /retrain permet de réentraîner le modèle à chaud avec de nouveaux hyperparamètres, sans redémarrer le serveur.

Le dashboard intégré (GET /dashboard) fournit une interface web complète pour contrôler tout le projet : statut des services, lancement des flows, prédiction interactive, réentraînement, et visualisation des métriques Elasticsearch.

---

## Étape 5 : MLflow (Atelier 5)

MLflow est intégré dans trainer.py et evaluator.py. Chaque exécution du pipeline crée un run MLflow qui enregistre les hyperparamètres (mlflow.log_param), les métriques (mlflow.log_metric), la matrice de confusion (mlflow.log_artifact) et le modèle complet (mlflow.sklearn.log_model).

Le tracking utilise un serveur MLflow local avec une base SQLite et un stockage d'artefacts sur le filesystem. L'interface web (port 5000) permet de comparer les runs côte à côte, visualiser l'évolution des métriques et télécharger les modèles.

---

## Étape 6 : Conteneurisation Docker (Atelier 6)

Le Dockerfile crée une image basée sur python:3.12-slim. Il copie les fichiers Python, le modèle entraîné et le dataset, installe les dépendances, et lance l'API FastAPI avec uvicorn.

Le .dockerignore exclut les fichiers inutiles (venv, __pycache__, mlflow.db) pour garder l'image légère.

L'image est nommée oussema_benachour_3asi_mlops selon la convention de l'atelier et poussée sur Docker Hub (obenachour/oussema_benachour_3asi_mlops) pour être partagée et déployable sur n'importe quelle machine.

Le Makefile automatise toutes les tâches : build, run, stop, tag, push, ainsi que les commandes du pipeline.

---

## Étape 7 : Monitoring Elasticsearch + Kibana (Atelier 7)

Un docker-compose.yml déploie Elasticsearch 7.17.9 et Kibana 7.17.9 en conteneurs Docker connectés sur un réseau commun.

Le fichier monitoring.py se connecte à Elasticsearch et envoie les métriques de chaque run (nom, paramètres, métriques, timestamp) vers l'index mlflow-metrics.

Dans Kibana (port 5601), on crée un index pattern mlflow-metrics avec le champ timestamp, puis on utilise Discover pour explorer les logs en temps réel et Dashboard pour créer des visualisations.

---

## Étape 8 : Dashboard de contrôle

Le dashboard web intégré dans app.py (accessible sur /dashboard) centralise le contrôle de tout le projet :

- Liens directs vers toutes les interfaces (Prefect, MLflow, Swagger, Kibana, Elasticsearch)
- Statut en temps réel de chaque service (vert = running, rouge = stopped)
- Boutons pour lancer chaque flow Prefect avec log en direct
- Formulaire de prédiction interactif avec résultat visuel
- Formulaire de réentraînement avec affichage de l'accuracy
- Tableau des dernières métriques récupérées depuis Elasticsearch

---

## Résumé

Le projet implémente un pipeline MLOps complet : le code modulaire est orchestré par Prefect, le modèle est exposé via FastAPI, les expériences sont suivies par MLflow, l'application est conteneurisée avec Docker, le monitoring est assuré par Elasticsearch et Kibana, et un dashboard web centralise le contrôle de l'ensemble.
