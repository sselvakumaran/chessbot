CXX ?= c++
CXXFLAGS ?= -std=c++20 -O2 -shared -fPIC

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

.PHONY: all clean train FORCE
all: $(LIB)

# usage: make train CONFIG=v4_quiescence
# caffeinate keeps the Mac awake for the duration of the run (-i idle,
# -s system sleep while on AC power)
train: $(LIB)
	caffeinate -is venv/bin/python -m train $(CONFIG)

$(UNITY): FORCE
	@mkdir -p $(BUILD_DIR)
	@{ echo '#include "../$(ENGINE_DIR)/bindings.cpp"'; \
		for f in $(MODULES); do echo "#include \"../$$f\""; done; } > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@

$(LIB): $(UNITY) $(ENGINE) $(MODULES)
	mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) -I src $(UNITY) -o $@

clean:
	rm -f $(LIB_DIR)/libchess.dylib $(LIB_DIR)/libchess.so $(BUILD_DIR)
