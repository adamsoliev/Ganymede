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

LINKER_SCRIPT=virt.ld
KERNEL_IMAGE=kernel

OBJS = \
	entry.o \
	kernel.o \
	uart.o \
	printf.o \

# QEMU
QEMU=qemu-system-riscv64
MACH=virt
MEM=128M
RUN=$(QEMU) -machine $(MACH) -bios none -kernel $(KERNEL_IMAGE) -m $(MEM) -nographic 

# QEMU (debug)
GDB_PORT=1234

all: $(KERNEL_IMAGE)

$(KERNEL_IMAGE): $(OBJS) defs.h
	$(LD) $(LDFLAGS) -T virt.ld -o $@ $(OBJS) 
	$(OBJDUMP) -S kernel > kernel.asm

%.o: %.c defs.h 
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.S defs.h 
	$(CC) $(CFLAGS) -c $< -o $@


_%.o: _%.c 
	$(CC) $(CFLAGS) -c $< -o $@

qemu: all 
	$(RUN)

clean:
	rm -f $(OBJS) $(KERNEL_IMAGE) *.asm *.d *.o $(UPROG)

debug: all
	@echo "*** Now run 'gdb-multiarch $(KERNEL_IMAGE)' in another window"
	@echo "*** Once in, run 'target remote localhost:$(GDB_PORT)'"
	$(RUN) -gdb tcp::$(GDB_PORT) -S
