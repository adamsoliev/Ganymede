CFLAGS=-std=c11 -g -fno-common
SRCS=$(wildcard *.c)
OBJS=$(addprefix build/,$(notdir $(SRCS:.c=.o)))
EXECUTABLE=build/baikalc

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

build/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJS): baikalc.h | build

build:
	mkdir -p build

test: $(EXECUTABLE)
	./test.sh

clean:
	rm -rf build

.PHONY: all test clean