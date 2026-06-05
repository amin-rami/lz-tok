CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?=

BIN := build/lztok
SRC := src/lztok.c

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC) | build
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

build:
	mkdir -p build

clean:
	rm -rf build
