CC=clang-15
CFLAGS=-Wall -Wpedantic
SRCS=$(wildcard *.c)
OBJS=$(addprefix build/,$(notdir $(SRCS:.c=.o)))
EXECUTABLE=build/ganymede

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

build/%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJS): ganymede.h | build

build:
	mkdir -p build

test: $(EXECUTABLE)
	./test.sh

clean:
	rm -rf build

.PHONY: all test clean