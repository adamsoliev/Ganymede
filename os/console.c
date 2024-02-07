#include <stdarg.h>
#include "defs.h"

static void printint(int xx, int base, int sign);
static void printptr(unsigned long x);

int console_read(char *buf, int len) {
        for (int i = 0; i < len; i++) {
                while (1) {
                        int c = uartgetc();
                        if (c != -1) {
                                buf[i] = c;
                                if (c == '\r') { /* Enter */
                                        buf[i] = 0;
                                        printf("\r\n");
                                        return i;
                                } else if (c == 0x03) { /* Ctrl + C */ 
                                        printf("Ctrl + C\r\n");
                                        return 0;
                                } else if (c == 0x7f) { /* Backspace */
                                        c = 0;
                                        if (i) printf("\b \b");
                                        i = i ? i - 2 : i - 1;
                                } else {
                                        printf("%c", c);
                                }
                                break;
                        }
                }
        }
        return len - 1;
}

void console_write(char *buf, int len) {
        for (int i = 0; i < len; i++) uartputc(buf[i]);
}

void console_init() {
        uartinit();
}

// should be moved out eventually
static char digits[] = "0123456789abcdef";

static void printint(int xx, int base, int sign) {
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

static void printptr(unsigned long x) {
        int i;
        uartputc('0');
        uartputc('x');
        for (i = 0; i < (sizeof(unsigned long) * 2); i++, x <<= 4)
                uartputc(digits[x >> (sizeof(unsigned long) * 8 - 4)]);
}

// only understands %d, %x, %p, %s
void printf(char *fmt, ...) {
        va_list ap;
        int i, c;
        char *s;

        if (fmt == 0) printf("null fmt");

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
                                // Print unknown % sequence to draw attention.
                                uartputc('%');
                                uartputc(c);
                                break;
                }
        }
        va_end(ap);
}
