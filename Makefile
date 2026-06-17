CXX ?= c++
CXXFLAGS ?= -std=c++20 -O2 -shared -fPIC

ENGINE_DIR := src/engine
LIB_DIR := lib
SRC := $(ENGINE_DIR)/bindings.cpp
DEPS := $(SRC) $(ENGINE_DIR)/game.cpp $(ENGINE_DIR)/engine.cpp

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LIB := $(LIB_DIR)/libchess.dylib
else
  LIB := $(LIB_DIR)/libchess.so
endif

.PHONY: all clean
all: $(LIB)

$(LIB): $(DEPS)
	mkdir -p $(LIB_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

clean:
	rm -f $(LIB_DIR)/libchess.dylib $(LIB_DIR)/libchess.so