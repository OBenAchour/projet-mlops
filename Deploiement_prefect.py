from prefect import serve
from pipeline_prefect import (
    all_flow,
    install_flow,
    train_flow,
    evaluate_flow,
    clean_flow,
    predict_flow,
    code_quality_flow,
    test_flow,
)

if __name__ == "__main__":

    # Déploiement du pipeline complet - schedulé tous les jours à 2h du matin
    all_deploy = all_flow.to_deployment(
        name="ml-pipeline-all",
        cron="0 2 * * *",
        tags=["full-pipeline", "mlops"],
    )

    # Déploiement installation
    install_deploy = install_flow.to_deployment(
        name="ml-pipeline-install",
        tags=["mlops", "setup"],
    )

    # Déploiement entraînement
    train_deploy = train_flow.to_deployment(
        name="ml-pipeline-train",
        tags=["training", "mlops"],
    )

    # Déploiement évaluation
    evaluate_deploy = evaluate_flow.to_deployment(
        name="ml-pipeline-evaluate",
        tags=["evaluation", "mlops"],
    )

    # Déploiement prédiction
    predict_deploy = predict_flow.to_deployment(
        name="ml-pipeline-predict",
        tags=["prediction", "mlops"],
    )

    # Déploiement qualité du code
    code_deploy = code_quality_flow.to_deployment(
        name="ml-pipeline-code",
        tags=["code-quality", "mlops"],
    )

    # Déploiement tests
    test_deploy = test_flow.to_deployment(
        name="ml-pipeline-test",
        tags=["testing", "mlops"],
    )

    # Déploiement nettoyage
    clean_deploy = clean_flow.to_deployment(
        name="ml-pipeline-clean",
        tags=["maintenance", "mlops"],
    )

    # Lancer tous les déploiements (worker)
    serve(
        all_deploy,
        install_deploy,
        train_deploy,
        evaluate_deploy,
        predict_deploy,
        code_deploy,
        test_deploy,
        clean_deploy,
    )
