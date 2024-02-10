KERNEL_IMAGE=kernel

OBJS = \
	entry.o \
	start.o \
	main.o \
	uart.o \
	kalloc.o \
	vm.o \

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

all: $(KERNEL_IMAGE)

$(KERNEL_IMAGE): $(OBJS)
	$(LD) $(LDFLAGS) -T kernel.ld -o $@ $(OBJS) 
	$(OBJDUMP) -S kernel > kernel.asm

%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.S 
	$(CC) $(CFLAGS) -c $< -o $@


QEMU=qemu-system-riscv64
RUN=$(QEMU) -machine virt -cpu rv64 -smp 1 -bios none -kernel $(KERNEL_IMAGE) -m 17M -nographic 

qemu: all 
	$(RUN)

clean:
	rm -rf $(OBJS) $(KERNEL_IMAGE) *.asm *.d *.o

GDB_PORT=1234

debug: all
	@echo "*** Now run 'gdb-multiarch $(KERNEL_IMAGE)' in another window"
	@echo "*** Once in, run 'target remote localhost:$(GDB_PORT)'"
	$(RUN) -gdb tcp::$(GDB_PORT) -S