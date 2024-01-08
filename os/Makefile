# Build
CC=riscv64-unknown-linux-gnu-gcc
LD=riscv64-unknown-linux-gnu-ld
OBJDUMP=riscv64-unknown-linux-gnu-objdump
OBJCOPY=riscv64-unknown-linux-gnu-objcopy
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
	initcode.o \


# QEMU
QEMU=qemu-system-riscv64
MACH=virt
MEM=128M
RUN=$(QEMU) -nographic -machine $(MACH) -m $(MEM)
RUN+=-bios none -kernel $(KERNEL_IMAGE)

# QEMU (debug)
GDB_PORT=1234

all: $(KERNEL_IMAGE)

$(KERNEL_IMAGE): $(OBJS) initcode
	$(LD) $(LDFLAGS) -T virt.ld -o $@ $(OBJS) 
	$(OBJDUMP) -S kernel > kernel.asm

initcode: initcode.S
	$(CC) $(CFLAGS) -march=rv64g -nostdinc -I. -c initcode.S -o initcode.o
	$(LD) $(LDFLAGS) -N -e start -Ttext 0 -o initcode.out initcode.o
	$(OBJCOPY) -S -O binary initcode.out initcode

%.o: %.c kernel.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.S kernel.h
	$(CC) $(CFLAGS) -c $< -o $@

qemu: all
	$(RUN)

clean:
	rm -f $(OBJS) $(KERNEL_IMAGE) *.asm initcode initcode.out *.d

debug: all
	@echo "*** Now run 'gdb-multiarch $(KERNEL_IMAGE)' in another window"
	@echo "*** Once in, run 'target remote localhost:$(GDB_PORT)'"
	$(RUN) -gdb tcp::$(GDB_PORT) -S
