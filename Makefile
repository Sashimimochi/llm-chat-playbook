setup:
	sh setup.sh

launch:
	docker compose up -d
	open http://localhost:8503

all:
	@make setup
	@make launch

down:
	docker compose down

clean:
	@make down
	docker system prune -f
	rm -rf ./model ./data ./logs ./vector_store
