// long long int x = 2;

// signed char c;
// signed short s;
// signed int i;
// signed long int l;
// unsigned char C;
// unsigned short S;
// unsigned int I;
// unsigned long int L;
// float f;
// double d;
// long double D;
// int b;
const int a, *x;
// int b, *y;
// volatile unsigned z;
// void *p;
// void (*P)(void);

// int i, j;
// int i, j, *p;
// float f[128];
// int in[] = {10, 32, -1, 567, 3, 18, 1, -51, 789, 0};
// int x[3][4], *y[3];
// int up[15], down[15], rows[8], x[8];
// int queens(), print();

int main(void) {
        unsigned long int a = 23;
        for (a = 0; a < 23; a++) {
                if (a == 10) continue;
                if (a == 16) break;
        }

L1:
        return 23;

        if (a == 100) goto L1;
        return 0;
}
