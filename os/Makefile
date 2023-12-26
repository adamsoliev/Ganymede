# Build
CC=riscv64-unknown-linux-gnu-gcc
LD=riscv64-unknown-linux-gnu-ld
OBJDUMP=riscv64-unknown-linux-gnu-objdump
CFLAGS=-Wall -Wextra -ffreestanding -nostartfiles -nostdlib -nodefaultlibs
CFLAGS+=-g -Wl,--gc-sections -mcmodel=medany
LDFLAGS = -z max-page-size=4096

LINKER_SCRIPT=virt.ld
KERNEL_IMAGE=kernel

OBJS = \
	entry.o \
	kernel.o

# QEMU
QEMU=qemu-system-riscv64
MACH=virt
MEM=128M
RUN=$(QEMU) -nographic -machine $(MACH) -m $(MEM)
RUN+=-bios none -kernel $(KERNEL_IMAGE)

# QEMU (debug)
GDB_PORT=1234

all: $(KERNEL_IMAGE)

$(KERNEL_IMAGE): $(OBJS)
	$(LD) $(LDFLAGS) -T virt.ld -o $@ $(OBJS) 
	$(OBJDUMP) -S kernel > kernel.asm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.S
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(KERNEL_IMAGE) */*.asm

debug: all
	@echo "*** Now run 'gdb-multiarch $(KERNEL_IMAGE)' in another window"
	@echo "*** Once in, run 'target remote localhost:$(GDB_PORT)'"
	$(RUN) -gdb tcp::$(GDB_PORT) -S
