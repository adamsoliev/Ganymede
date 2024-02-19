K=kernel
U=user

KERNEL_IMAGE=kernel

OBJS = \
	$K/entry.o \
	$K/start.o \
	$K/main.o \
	$K/uart.o \
	$K/kalloc.o \
	$K/vm.o \
	$K/proc.o \
	$K/trap.o \
	$K/trapvecs.o \
	$K/swtch.o \
	$K/printf.o \
	# $K/trampoline.o \


ifndef TOOLPREFIX
TOOLPREFIX := $(shell if riscv64-unknown-elf-objdump -i 2>&1 | grep 'elf64-big' >/dev/null 2>&1; \
	then echo 'riscv64-unknown-elf-'; \
	elif riscv64-linux-gnu-objdump -i 2>&1 | grep 'elf64-big' >/dev/null 2>&1; \
	then echo 'riscv64-linux-gnu-'; \
	elif riscv64-unknown-linux-gnu-objdump -i 2>&1 | grep 'elf64-big' >/dev/null 2>&1; \
	then echo 'riscv64-unknown-linux-gnu-'; \
	else echo "***" 1>&2; \
	echo "*** Error: Couldn't find a riscv64 version of GCC/binutils." 1>&2; \
	echo "*** To turn off this error, run 'gmake TOOLPREFIX= ...'." 1>&2; \
	echo "***" 1>&2; exit 1; fi)
endif

CC=$(TOOLPREFIX)gcc
CFLAGS = -Wall -Werror -fno-omit-frame-pointer -ggdb -gdwarf-2
CFLAGS += -MD -mcmodel=medany -I.
CFLAGS += -ffreestanding -fno-common -nostdlib -mno-relax
LD=$(TOOLPREFIX)ld
LDFLAGS = -z max-page-size=4096
OBJDUMP=$(TOOLPREFIX)objdump
OBJCOPY=$(TOOLPREFIX)objcopy

all: $K/$(KERNEL_IMAGE)

$K/$(KERNEL_IMAGE): $(OBJS) $U/proc1 $U/proc2
	$(LD) $(LDFLAGS) -T $K/kernel.ld -o $@ $(OBJS) 
	$(OBJDUMP) -S $K/kernel > $K/kernel.asm

$U/proc1: $U/proc1.S
	$(CC) $(CFLAGS) -march=rv64g -nostdinc -c $U/proc1.S -o $U/proc1.o
	$(LD) $(LDFLAGS) -N -e start -Ttext 0 -o $U/proc1.out $U/proc1.o
	$(OBJCOPY) -S -O binary $U/proc1.out $U/proc1
	$(OBJDUMP) -S $U/proc1.o > $U/proc1.asm

$U/proc2: $U/proc2.S
	$(CC) $(CFLAGS) -march=rv64g -nostdinc -c $U/proc2.S -o $U/proc2.o
	$(LD) $(LDFLAGS) -N -e start -Ttext 0 -o $U/proc2.out $U/proc2.o
	$(OBJCOPY) -S -O binary $U/proc2.out $U/proc2
	$(OBJDUMP) -S $U/proc2.o > $U/proc2.asm

%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.S 
	$(CC) $(CFLAGS) -c $< -o $@

mkfs/mkfs: mkfs/mkfs.c $K/types.h
	@gcc -Werror -Wall -I. -o mkfs/mkfs mkfs/mkfs.c
	@./mkfs/mkfs fs.img $U/proc1 $U/proc2

QEMU=qemu-system-riscv64
RUN=$(QEMU) -machine virt -cpu rv64 -smp 1 -bios none -kernel $K/$(KERNEL_IMAGE) -m 17M -nographic 

qemu: all 
	$(RUN)

clean:
	rm -rf $(OBJS) $K/$(KERNEL_IMAGE) */*.asm */*.d */*.o */*.out \
	$U/proc1 $U/proc2 mkfs/mkfs fs.img

GDB_PORT=1234

debug: all
	@echo "*** Now run 'gdb-multiarch $K/$(KERNEL_IMAGE)' in another window"
	@echo "*** Once in, run 'target remote localhost:$(GDB_PORT)'"
	$(RUN) -gdb tcp::$(GDB_PORT) -S