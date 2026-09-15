setup:
	sh setup.sh

launch:
	docker-compose up -d ollama llm
	docker-compose run --rm ollama-init
	cp ~/.gitconfig .gitconfig
	open http://localhost:8503
	docker-compose run --rm opencode

all:
	@make setup
	@make launch

down:
	docker compose down

clean:
	@make down
	docker system prune -f
	rm -rf ./model ./data ./logs ./vector_store
