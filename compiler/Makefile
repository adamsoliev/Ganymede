CFLAGS=-std=c11 -g -fno-common
SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.o)

baikalc: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) 

$(OBJS): baikalc.h

test: baikalc
	./test.sh

clean: 
	rm -f baikalc *.o *~ tmp*

.PHONY: test clean