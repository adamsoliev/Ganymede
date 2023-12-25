# Build
CC=riscv64-unknown-linux-gnu-gcc
CFLAGS=-Wall -Wextra -ffreestanding -nostartfiles -nostdlib -nodefaultlibs
CFLAGS+=-g -Wl,--gc-sections -mcmodel=medany
RUNTIME=entry.S
LINKER_SCRIPT=virt.ld
KERNEL_IMAGE=main

# QEMU
QEMU=qemu-system-riscv64
MACH=virt
MEM=128M
RUN=$(QEMU) -nographic -machine $(MACH) -m $(MEM)
RUN+=-bios none -kernel $(KERNEL_IMAGE)

# QEMU (debug)
GDB_PORT=1234

all: main
	$(CC) $(CFLAGS) $(RUNTIME) *.o -T $(LINKER_SCRIPT) -o $(KERNEL_IMAGE)

main:
	$(CC) -c kernel.c $(CFLAGS) -o main.o

run: clean all
	$(RUN)

debug: all
	@echo "*** Now run 'gdb-multiarch $(KERNEL_IMAGE)' in another window"
	@echo "*** Once in, run 'target remote localhost:$(GDB_PORT)'"
	$(RUN) -gdb tcp::$(GDB_PORT) -S

clean:
	rm -vf *.o
	rm -vf $(KERNEL_IMAGE)
	rm .gdb_history
