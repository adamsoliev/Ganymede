#include "defs.h"

int main(void) {
        console_init();

        while (1) {
                for (int i = 0; i < 1000000; i++)
                        ;
                printf("hello\n");
        }

        return 0;
}
