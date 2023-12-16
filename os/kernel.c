#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

__attribute__((aligned(512))) char _stack[4096];

////////////////
// CONSOLE (& uart)
////////////////
#define UART0 0x10000000L
#define Reg(reg) ((volatile unsigned char *)(UART0 + reg))

/* clang-format off */
// see https://www.lammertbies.nl/comm/info/serial-uart
#define RHR                     0               // receive holding register (for input bytes)
#define THR                     0               // transmit holding register (for output bytes)
#define IER                     1               // interrupt enable register
#define IER_RX_ENABLE           (1<<0)
#define IER_TX_ENABLE           (1<<1)
#define FCR                     2               // FIFO control register
#define FCR_FIFO_ENABLE         (1<<0)
#define FCR_FIFO_CLEAR          (3<<1)          // clear the content of the two FIFOs
#define ISR                     2               // interrupt status register
#define LCR                     3               // line control register
#define LCR_EIGHT_BITS          (3<<0)
#define LCR_BAUD_LATCH          (1<<7)          // special mode to set baud rate
#define LSR                     5               // line status register
#define LSR_RX_READY            (1<<0)          // input is waiting to be read from RHR
#define LSR_TX_IDLE             (1<<5)          // THR can accept another character to send

#define ReadReg(reg) (*(Reg(reg)))
#define WriteReg(reg, v) (*(Reg(reg)) = (v))
/* clang-format on */

void uartinit(void) {
        WriteReg(IER, 0x00);            // disable interrupts.
        WriteReg(LCR, LCR_BAUD_LATCH);  // special mode to set baud rate.
        WriteReg(0, 0x03);              // LSB for baud rate of 38.4K.
        WriteReg(1, 0x00);              // MSB for baud rate of 38.4K.
        WriteReg(LCR, LCR_EIGHT_BITS);  // and set word length to 8 bits, no parity.
        WriteReg(FCR, FCR_FIFO_ENABLE | FCR_FIFO_CLEAR);  // reset and enable FIFOs.
        WriteReg(IER, IER_TX_ENABLE | IER_RX_ENABLE);     // enable transmit and receive interrupts.
}

void uartputc(int c) { WriteReg(0, c); }

int uartgetc(void) {
        if (ReadReg(5) & 0x01) return ReadReg(0);
        return -1;
}

////////////////
// PRINTS
////////////////
static char digits[] = "0123456789abcdef";
void printint(int xx, int base, int sign) {
        char buf[16];
        int i;
        unsigned int x;
        if (sign && (sign = xx < 0))
                x = -xx;
        else
                x = xx;

        i = 0;
        do {
                buf[i++] = digits[x % base];
        } while ((x /= base) != 0);

        if (sign) buf[i++] = '-';
        while (--i >= 0) uartputc(buf[i]);
}

void printptr(unsigned long x) {
        int i;
        uartputc('0');
        uartputc('x');
        for (i = 0; i < (int)(sizeof(unsigned long) * 2); i++, x <<= 4)
                uartputc(digits[x >> (sizeof(unsigned long) * 8 - 4)]);
}

void printf(char *fmt, ...) {
        va_list ap;
        int i, c;
        char *s;
        if (fmt == 0) {
                uartputc('n');
                uartputc('u');
                uartputc('l');
                uartputc('l');
        }

        va_start(ap, fmt);
        for (i = 0; (c = fmt[i] & 0xff) != 0; i++) {
                if (c != '%') {
                        uartputc(c);
                        continue;
                }
                c = fmt[++i] & 0xff;
                if (c == 0) break;
                switch (c) {
                        case 'd': printint(va_arg(ap, int), 10, 1); break;
                        case 'x': printint(va_arg(ap, int), 16, 1); break;
                        case 'p': printptr(va_arg(ap, unsigned long)); break;
                        case 's':
                                if ((s = va_arg(ap, char *)) == 0) s = "(null)";
                                for (; *s; s++) uartputc(*s);
                                break;
                        case '%': uartputc('%'); break;
                        default:
                                uartputc('%');
                                uartputc(c);
                                break;
                }
        }
        va_end(ap);
}

////////////////
// RISC-V
////////////////
#define PGSIZE 4096  // bytes per page
#define PGSHIFT 12   // bits of offset within a page
#define PGROUNDUP(sz) (((sz) + PGSIZE - 1) & ~(PGSIZE - 1))
#define PGROUNDDOWN(a) (((a)) & ~(PGSIZE - 1))

////////////////
// MEMLAYOUT
////////////////
#define KERNBASE 0x80000000L
#define PHYSTOP (KERNBASE + 128 * 1024 * 1024)  // 128 MB

char end[];  // first address after kernel
             // defined by virt.ld

struct run {
        struct run *next;
};

struct {
        struct run *freelist;
} kmem;

void kinit() {
        char *p = (char *)PGROUNDUP((unsigned long)end);
        for (; p + PGSIZE <= (char *)(void *)PHYSTOP; p += PGSIZE) {
                // add a page of physical memory pointed at by p to head of list
                struct run *r;
                r = (struct run *)p;
                r->next = kmem.freelist;
                kmem.freelist = r;
        }
}

// allocate one 4096-byte page of physical memory
void *kalloc(void) {
        struct run *r;
        r = kmem.freelist;
        if (r) kmem.freelist = r->next;
        return (void *)r;
}

void kfree(void *pa) {
        struct run *r;
        r = (struct run *)pa;
        r->next = kmem.freelist;
        kmem.freelist = r;
}

void main(void) {
        printf("%s", "------------------------------------\r\n");
        printf("%s", "<<<      64-bit RISC-V OS        >>>\r\n");
        printf("%s", "------------------------------------\r\n");

        uartinit();
        kinit();

        unsigned long *page = kalloc();
        printf("allocated page: %p\r\n", page);
        kfree(page);
        unsigned long *page1 = kalloc();
        printf("allocated page1: %p\r\n", page1);

        while (1) {
                int c = uartgetc();
                if (c != -1) {
                        if (c == '\r')
                                uartputc('\n');
                        else if (c == 0x7f)
                                printf("%s", "\b \b");
                        else
                                uartputc(c);
                }
        }
}
