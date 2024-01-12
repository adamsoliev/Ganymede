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
RUN=$(QEMU) -machine $(MACH) -bios none -kernel $(KERNEL_IMAGE) -m $(MEM) -nographic 
# RUN+= -global virtio-mmio.force-legacy=false
RUN+= -drive file=fs.img,if=none,format=raw,id=x0

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

%.o: %.c kernel.h defs.h param.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.S kernel.h defs.h param.h
	$(CC) $(CFLAGS) -c $< -o $@

ULIB = usys.o 

_%.o: _%.c 
	$(CC) $(CFLAGS) -c $< -o $@

_%: _%.o $(ULIB)
	$(LD) $(LDFLAGS) -T user.ld -o $@ $^
	$(OBJDUMP) -S $@ > $*.asm

usys.o: usys.S
	$(CC) $(CFLAGS) -c -o usys.o usys.S

mkfs/mkfs: mkfs/mkfs.c kernel.h param.h
	gcc -Werror -Wall -I. -o mkfs/mkfs mkfs/mkfs.c

UPROG=\
	_sh\

fs.img: mkfs/mkfs $(UPROG)
	mkfs/mkfs fs.img $(UPROG)

qemu: all fs.img
	$(RUN)

clean:
	rm -f $(OBJS) $(KERNEL_IMAGE) *.asm initcode initcode.out *.d *.o $(UPROG) \
	mkfs/mkfs fs.img

debug: all fs.img
	@echo "*** Now run 'gdb-multiarch $(KERNEL_IMAGE)' in another window"
	@echo "*** Once in, run 'target remote localhost:$(GDB_PORT)'"
	$(RUN) -gdb tcp::$(GDB_PORT) -S
