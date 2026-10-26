# Common development tasks. Run `make install` inside an activated virtual environment.
.PHONY: install lint test video

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

test:
	pytest

# Renders the vertical promo video (needs Node.js 22+, FFmpeg, and network access
# on first run to fetch the HyperFrames CLI and headless Chrome).
video:
	cd promo/video && npx --yes hyperframes@0.8.206 browser ensure
	cd promo/video && npx --yes hyperframes@0.8.206 render --quality delivery --fps 30 --output out/devwaste-linkedin-vertical.mp4
