CONFIG ?= springs_small

.PHONY: data train verify test lint

test:
	python -m pytest -q

lint:
	python -m ruff check fabric tests

data:
	@echo "Not yet implemented (M1). Will invoke: python scripts/generate_data.py --config=$(CONFIG)"
	@exit 1

train:
	@echo "Not yet implemented (M4). Will invoke: python scripts/train.py --config=$(CONFIG)"
	@exit 1

verify:
	@echo "Not yet implemented (M5). Will invoke: python scripts/verify_emergence.py --config=$(CONFIG)"
	@exit 1
