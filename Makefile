REAL_USER := $(if $(SUDO_USER),$(SUDO_USER),$(USER))
REAL_HOME := $(shell getent passwd $(REAL_USER) | cut -d: -f6)

setup:
	bash setup.sh
	cp .env.example .env

install-skills:
	git clone https://github.com/farmage/opencode-skills.git
	bash opencode-skills/install.sh --local
	rm -rf opencode-skills

launch:
	docker compose up -d ollama llm
	docker compose run --rm ollama-init
	cp $(REAL_HOME)/.gitconfig .gitconfig
	open http://localhost:8053
	docker compose run --rm opencode

all:
	@make setup
	@make install-skills
	@make launch

down:
	docker compose down --remove-orphans

clean:
	@make down
	docker system prune -f
	rm -rf ./model ./data ./logs ./vector_store
