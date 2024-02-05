K=kernel

# Build
CC=riscv64-unknown-elf-gcc
LD=riscv64-unknown-elf-ld
OBJDUMP=riscv64-unknown-elf-objdump
OBJCOPY=riscv64-unknown-elf-objcopy
CFLAGS = -Wall -Werror -O -fno-omit-frame-pointer -ggdb -gdwarf-2
CFLAGS += -MD
CFLAGS += -mcmodel=medany
CFLAGS += -ffreestanding -fno-common -nostdlib -mno-relax
CFLAGS += -I.
LDFLAGS = -z max-page-size=4096

LINKER_SCRIPT=kernel.ld
KERNEL_IMAGE=kernel

OBJS = \
	$K/entry.o \
	$K/start.o \
	$K/kernel.o \
	$K/uart.o \
	$K/printf.o \
	$K/console.o \

# QEMU
QEMU=qemu-system-riscv64
MACH=virt
MEM=128M
RUN=$(QEMU) -machine $(MACH) -bios none -kernel $K/$(KERNEL_IMAGE) -m $(MEM) -nographic 

# QEMU (debug)
GDB_PORT=1234

all: $K/$(KERNEL_IMAGE)

$K/$(KERNEL_IMAGE): $(OBJS) $K/defs.h
	$(LD) $(LDFLAGS) -T $K/kernel.ld -o $@ $(OBJS) 
	$(OBJDUMP) -S $K/kernel > $K/kernel.asm

%.o: %.c $K/defs.h 
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.S $K/defs.h 
	$(CC) $(CFLAGS) -c $< -o $@

qemu: all 
	$(RUN)

clean:
	rm -rf $(OBJS) $K/$(KERNEL_IMAGE) $K/*.asm $K/*.d $K/*.o

debug: all
	@echo "*** Now run 'gdb-multiarch $(KERNEL_IMAGE)' in another window"
	@echo "*** Once in, run 'target remote localhost:$(GDB_PORT)'"
	$(RUN) -gdb tcp::$(GDB_PORT) -S
