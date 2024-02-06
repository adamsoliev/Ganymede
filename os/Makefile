
KERNEL_IMAGE=kernel

OBJS = \
	entry.o \
	main.o \
	uart.o \

CC=riscv64-unknown-elf-gcc
CFLAGS = -Wall -Werror -fno-omit-frame-pointer -ggdb -gdwarf-2
CFLAGS += -MD -mcmodel=medany -I.
CFLAGS += -ffreestanding -fno-common -nostdlib -mno-relax
LD=riscv64-unknown-elf-ld
LDFLAGS = -z max-page-size=4096
OBJDUMP=riscv64-unknown-elf-objdump

all: $(KERNEL_IMAGE)

$(KERNEL_IMAGE): $(OBJS)
	$(LD) $(LDFLAGS) -T kernel.ld -o $@ $(OBJS) 
	$(OBJDUMP) -S kernel > kernel.asm

%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.S 
	$(CC) $(CFLAGS) -c $< -o $@


QEMU=qemu-system-riscv64
RUN=$(QEMU) -machine virt -cpu rv64 -smp 1 -s -bios none -kernel $(KERNEL_IMAGE) -m 128M -nographic 

qemu: all 
	$(RUN)

clean:
	rm -rf $(OBJS) $(KERNEL_IMAGE) *.asm *.d *.o
