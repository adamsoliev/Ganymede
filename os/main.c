// https://www.lammertbies.nl/comm/info/serial-uart
#define UART0 0x10000000L
#define IER 1  // interrupt enable
#define FCR 2  // fifo control
#define LCR 3  // line control
#define LSR 5  // line status

#define Reg(reg) ((volatile unsigned char *)(UART0 + reg))

void uartinit() {
        (*(Reg(IER))) = 0x00;      // turn off interrupts
        (*(Reg(LCR))) = (1 << 7);  // access rate setter registers
        (*(Reg(0))) = 0x01;        // DLL
        (*(Reg(1))) = 0x00;        // DLM
        (*(Reg(LCR))) = 0x11;      // data size 8-bit
        (*(Reg(FCR))) = (unsigned char) 0x111;     // clear Tx FIFO | clear Rx FIFO | Enable FIFO
        (*(Reg(IER))) = 0x11;      // enable Tx | enable Rx
}

void uartputc(int c) {
        while (((*(Reg(LSR))) & (1 << 5)) == 0)
                ;
        *(Reg(0)) = c;
}

int uartgetc(void) {
        if ((*(Reg(LSR))) & 0x1) {
                return *(Reg(0));
        } else
                return -1;
}

__attribute__ ((aligned (128))) char stack0[4096];
int main(void);

void start() {
        // set prev to supervisor
        unsigned long mstatus;
        asm volatile("csrr %0, mstatus" : "=r"(mstatus));
        asm volatile("csrw mstatus, %0" ::"r"((mstatus & ~(3L << 11)) | (1L << 11)));

        // supervisor entry
        asm volatile("csrw mepc, %0" ::"r"(main));

        // configure physical memory protection to give supervisor access to all memory
        asm volatile("csrw pmpaddr0, %0" ::"r"(0x3fffffffffffffULL));
        asm volatile("csrw pmpcfg0, %0" ::"r"(0xf));

        // switch to supervisor
        asm volatile("mret");
}

int main(void) {
        uartinit();

        uartputc('h');
        uartputc('e');
        uartputc('l');
        uartputc('l');
        uartputc('o');
        uartputc('\n');

        return 0;
}
