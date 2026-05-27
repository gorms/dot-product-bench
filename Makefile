CC      = gcc
CFLAGS  = -O2 -Wall -Wextra

bench: main.c
	$(CC) $(CFLAGS) -o bench main.c

clean:
	rm -f bench

.PHONY: clean
