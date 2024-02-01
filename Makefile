DOCKER_BUILD_AND_RUN_IN_BACKGROUND = docker-compose -f docker-compose.yml -f docker-compose.local.yml up --build -d
PYSTRODYNAMICS_CONTAINER_NAME = pystrodynamics-pystrodynamics-1
EXECUTE_IN_PYSTRODYNAMICS_CONTAINER = docker exec -it $(PYSTRODYNAMICS_CONTAINER_NAME)

build:
	$(DOCKER_BUILD_AND_RUN_IN_BACKGROUND)

unit_tests:
	$(EXECUTE_IN_PYSTRODYNAMICS_CONTAINER) python3 -m pytest tests/unit_tests/