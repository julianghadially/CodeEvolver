# Copyright © 2026 440 Labs LLC
# Thin wrapper over orchestrator/build.py, which is driven by CE_BENCHMARKS and
# CE_REPO_ROOT / CE_REPO_<SLUG>. No researcher's home directory is baked in here
# -- teammates running different benchmark subsets use this unchanged.
#
#   export CE_BENCHMARKS=hover,phantomwiki,ragqa
#   make bundles && make images && make push
#
# Local (Apple Silicon) build for the same images, into the local daemon:
#
#   make bundles
#   PLATFORM=linux/arm64 REGISTRY=local make base images
BUILD := python3 ../orchestrator/build.py

# Droplets are x86_64, so that stays the default. Override for a native build
# on an Apple-Silicon Mac; Dockerfile.base handles the one thing that differs
# between the two (MongoDB ships no arm64 package in its Debian repo).
PLATFORM ?= linux/amd64
REGISTRY ?= local

.PHONY: all bundles base images push show clean
all: bundles images

show:    ; @python3 ../orchestrator/fleet_config.py
bundles: ; $(BUILD) bundles
base:    ; $(BUILD) base   --platform $(PLATFORM)
images:  ; $(BUILD) images --platform $(PLATFORM) --registry $(REGISTRY)
push:    ; $(BUILD) images --platform $(PLATFORM) --registry $(REGISTRY) --push
clean:   ; rm -rf bundles
