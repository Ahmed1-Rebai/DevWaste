.PHONY: install lint test video

PROMO := promo/video
SFX_SRC ?= .agents/skills/media-use/audio/assets/sfx

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

test:
	pytest

# Renders the vertical LinkedIn promo (needs Node 22+ and FFmpeg).
# Sound effects come from the HyperFrames media-use skill:
#   npx skills add heygen-com/hyperframes --all --full-depth
video:
	cd $(PROMO) && npm install
	mkdir -p $(PROMO)/assets/sfx
	cp $(SFX_SRC)/whoosh.mp3 $(SFX_SRC)/whoosh-cinematic.mp3 \
		$(SFX_SRC)/notification.mp3 $(SFX_SRC)/impact-bass-2.mp3 $(PROMO)/assets/sfx/
	cd $(PROMO) && npx hyperframes check
	cd $(PROMO) && npx hyperframes render --quality delivery --fps 30 \
		--video-frame-format png --output out/devwaste-linkedin-vertical.mp4
