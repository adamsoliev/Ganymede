

all: kernel entry link run

kernel: 
	@riscv64-unknown-linux-gnu-gcc -Wall -Wextra -c -mcmodel=medany kernel.c -o kernel.o -ffreestanding

entry:
	@riscv64-unknown-linux-gnu-as -c entry.S -o entry.o

link:
	@riscv64-unknown-linux-gnu-ld -T linker.ld -nostdlib kernel.o entry.o -o kernel.elf

run: 
	@qemu-system-riscv64 -machine virt -bios none -kernel kernel.elf -serial mon:stdio

clean:
	@rm -vf *.o
	@rm -vf kernel.elf 
