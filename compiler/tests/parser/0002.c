int bb = 23 + 32 + 23 - 43 - 32;
int bc = 82 + 23 * 32 - 600 / 20;
int bd = 54 << 2;
int be = 24 >> 2;
int bf = 94 < 2;
int bg = 74 > 2;
int bh = 13 <= 2;
int bi = 94 >= 2;
int bj = 51 == 2;
int bk = 18 != 2;
int bl = 43 & 2;
int bm = 93 ^ 2;
int bn = 34 | 2;
int bo = 32 && 92;
int bp = 32 || 12;
int bv = sizeof bb;

int sum0() { return 0; }

int sum1(int a) { return a; }

int sum2(int a, int b) { return a + b; }

int sum10(int a, int b, int c, int d, int e, int f, int g, int h, int i) {
        return a + b + c + d + e + f + g + h + i;
}

int a[4] = {1, 2, 3, 9};
int b[3][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 0, 1, 2}};

int main() {
        float a = .2f;
        float a0 = .1250E+04f;
        float a1 = 1.250E3f;
        float a2 = 12.50E02f;
        float a3 = 125.0e+1f;
        float a4 = 1250e00f;
        float a5 = 12500.e-01f;
        float a6 = 125000e-2f;
        float a7 = 1250.f;
        double a8 = 1250.;
        double a9 = 12500.e-01;
        char abc = '\n';
        char str[10] = "hello";
        a = 32.94f;
        a9 = 120.34;
        int b = 42;
        int *ptr = &b;
        int **ptrPtr = &ptr;
        int array[5] = {1, 2, 3, 4, 5};
        int array1[5][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        printf(bb, bc);
        printf("a", b);
        printf("squared: %d\n", a * a);
        printf("escape char: %c\n", '\n');
        printf("escape char: %c\n", '\\');
        printf("escape char: %c\n", '\?');
        printf("escape char: %c\n", '\'');
        printf("escape char: %c\n", '\"');
        printf("normal char: %c\n", 'a');
        int aa = b == 23 ? 32 : 43;
        aa = b != 23 ? 32 : 43;
        int aaa = ++b;
        --aaa;
        ++aaa;
        aaa++;
        aaa--;
        int bbb;
        +bbb;
        -bbb;
        return 0;
}