CXX ?= c++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -shared -fPIC

ENGINE_DIR := src/engine
LIB_DIR := lib
BUILD_DIR := build

MODULES := $(sort $(wildcard src/search/*/*.cpp))
ENGINE := $(wildcard $(ENGINE_DIR)/*.cpp)
UNITY := $(BUILD_DIR)/unity.cpp

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LIB := $(LIB_DIR)/libchess.dylib
else
  LIB := $(LIB_DIR)/libchess.so
endif

.PHONY: all clean train tune FORCE
all: $(LIB)

# usage: make train CONFIG=v4_quiescence
# caffeinate keeps the Mac awake for the duration of the run (-i idle,
# -s system sleep while on AC power)
train: $(LIB)
	caffeinate -is venv/bin/python -m train $(CONFIG)

# usage: make tune NAME [-- FLAGS]     e.g.  make tune protein-2 -- --duty 0.5 --hours 9.5 --host m2-pro
# runs the Protein sweep configs/sweeps/NAME.json (default protein-2) until stopped (ctrl-c; the same command continues).
# FLAGS go to `python -m tune` unchanged; make itself rejects --flags, so they come after --
ifeq (tune,$(firstword $(MAKECMDGOALS)))
  TUNE_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  TUNE_NAME := $(if $(filter -%,$(firstword $(TUNE_ARGS))),,$(firstword $(TUNE_ARGS)))
  TUNE_FLAGS := $(if $(TUNE_NAME),$(wordlist 2,$(words $(TUNE_ARGS)),$(TUNE_ARGS)),$(TUNE_ARGS))
  $(eval $(TUNE_ARGS):;@:)
endif
tune: $(LIB)
	caffeinate -is venv/bin/python -m tune configs/sweeps/$(or $(TUNE_NAME),protein-2).json $(TUNE_FLAGS)

$(UNITY): FORCE
	@mkdir -p $(BUILD_DIR)
	@{ echo '#include "../$(ENGINE_DIR)/bindings.cpp"'; \
		for f in $(MODULES); do echo "#include \"../$$f\""; done; } > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@

$(LIB): $(UNITY) $(ENGINE) $(MODULES)
	mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) -I src $(UNITY) -o $@

clean:
	rm -rf $(LIB_DIR)/libchess.dylib $(LIB_DIR)/libchess.so $(BUILD_DIR)
