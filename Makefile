IMAGE_NAME = oussema_benachour_3asi_mlops
DOCKER_USER = oussemabenachour

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -d -p 8000:8000 --name ml-api $(IMAGE_NAME)

stop:
	docker stop ml-api && docker rm ml-api

tag:
	docker tag $(IMAGE_NAME) $(DOCKER_USER)/$(IMAGE_NAME):latest

push: tag
	docker push $(DOCKER_USER)/$(IMAGE_NAME):latest

login:
	docker login

api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000 &

train:
	python pipeline_prefect.py --flow train

evaluate:
	python pipeline_prefect.py --flow evaluate

all:
	python pipeline_prefect.py --flow all

test:
	pytest -v

quality:
	bash quality_check.sh
