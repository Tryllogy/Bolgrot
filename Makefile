SRC := src
UV := uv
ITERS := 300

all: run

install:
	$(UV) sync

run: install
	$(UV) run python -m $(SRC)

# PPO training / playback (requires the `ai` optional dependency group).
train: install
	$(UV) run --extra ai python -m $(SRC).ai.train --iterations $(ITERS)

# Continue training the last run's checkpoint (src/ai/ppo_ckpt.pt).
resume: install
	$(UV) run --extra ai python -m $(SRC).ai.train --iterations $(ITERS) --resume

play: install
	$(UV) run --extra ai python -m $(SRC).ai.play

lint:
	flake8 $(SRC)
	mypy $(SRC)

clean:
	find . -type d -name "__pycache__" | xargs rm -rf || true
	rm -rf .mypy_cache .pytest_cache build dist *.egg-info