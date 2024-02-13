#include <stdio.h>
#include <stdarg.h>

char digits[] = "0123456789abcdef";

void put(char c) {
  if (c == 0)
    printf("\nput: null\n");
  putc(c, stdout);
}

// supports positive numbers and 10, 16 bases
void printint(int xx, int base) {
  char buf[20];
  int i = 0;
  while (xx > 0) {
    int ld = xx % base;
    xx /= base;
    buf[i++] = digits[ld];
  }
  while (--i >= 0) {
    put(buf[i]);
  }
  
}

// supports %d, %x, %s
void printf_custom(char *fmt, ...) {
  va_list argptr;
  va_start(argptr, fmt);

  for (int i = 0; fmt[i] != 0; i++) {
    char c = fmt[i];
    if (c != '%') {
      put(c);
      continue;
    }
    c = fmt[++i];
    if (c == 'd' || c == 'x') {
      int x = va_arg(argptr, int);
      printint(x, c == 'd' ? 10 : 16);
    } else if (c == 's') {
      char *s = va_arg(argptr, char *);
      if (s == 0) s = "empty string";
      while (*s != '\0') put(*s++);
    } else if (c == 'c') {
      c = (char) va_arg(argptr, int);
      put(c);
    } else { 
      put('%');
      if (c == '%') continue;
      put(c);
    }
  }

  va_end(argptr);
}

int main() {
  printf_custom("0x%d\n", 29);
  printf("0x%d\n", 29);

  printf_custom("0x%x\n", 29);
  printf("0x%x\n", 29);

  printf_custom("%s\n", "hello");
  printf("%s\n", "hello");

  printf_custom("%%\n");
  printf("%%\n");

  printf_custom("%k\n");
  printf("%k\n");

  int num = 23;
  printf_custom("%c\n", num == 23 ? 'a' : 'b');
  printf("%c\n", num == 23 ? 'a' : 'b');

  return 0;
}

