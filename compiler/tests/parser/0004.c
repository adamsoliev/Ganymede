// /* NEGATIVE DECLSPECS TESTS */
// extern static int num = 23;
// int void num = 23;
// int char c = 'a';
// short char cc = 'a';
// long char cc = 'a';
// float int num = 32;
// short long int num = 32;
// int float fnum = 23.4f;
// char double fnum = 23.4f;
// short float shfl;
// long short int lshnum = 23;
// long char c = '\n';
// short long int shlnum = 32;
// signed float fnum = 12.3f;
// unsigned signed int num = 98;
// signed unsigned int num = 98;

/* TYPES TESTS */
struct MyStruct {
        char a;
        int b;
};

union MyUnion {
        char a;
        int b;
};

enum MyEnum { A, B, C };

// Basic Types
void a;
char b;
int c;
float d;
double e;
struct MyStruct f;
union MyUnion g;
enum MyEnum h;

// Basic Type + Compound Type: Pointer to T
int *a;
char *b;
struct MyStruct *c;
union MyUnion *d;
enum MyEnum *e;

// Basic Type + Compound Type: Array of T
int f[10];
char g[20];
struct MyStruct h[5];
union MyUnion i[5];
enum MyEnum j[5];

// Basic Type + Compound Type: Function returning T
int k();
char l();
struct MyStruct m();
union MyUnion n();
enum MyEnum o();

// Compound Type + Compound Type: Pointer to Pointer to T
int **p;
char **q;
struct MyStruct **r;
union MyUnion **s;
enum MyEnum **t;

// Compound Type + Compound Type: Pointer to Array of T
int (*u)[10];
char (*v)[20];
struct MyStruct (*w)[5];
union MyUnion (*x)[5];
enum MyEnum (*y)[5];

// Compound Type + Compound Type: Array of Pointer to T
int *z[10];
char *aa[20];
struct MyStruct *bb[5];
union MyUnion *cc[5];
enum MyEnum *dd[5];

// Compound Type + Compound Type: Pointer to Function returning T
int (*ee)();
char (*ff)();
struct MyStruct (*gg)();
union MyUnion (*hh)();
enum MyEnum (*ii)();

// Compound Type + Compound Type: Array of Function Pointers
int (*jj[5])();
char (*kk[5])();
struct MyStruct (*ll[5])();
union MyUnion (*mm[5])();
enum MyEnum (*nn[9])();
/* END */

int main() {
        /* CONSTANTS AND SYMBOL TABLE TESTS */
        int numa = 23;
        char *namea = "Hello World";
        char aa = '\a';
        char ba = '\b';
        char ca = '\f';
        char da = '\n';
        char ea = '\r';
        char fa = '\t';
        char ga = '\v';
        char ha = 'a';
        char ia = '\\';
        char ja = '\'';
        char ka = '\"';
        char la = '\?';
        float nfa = 2.3f;
        numa = 92;
        double PI = 3.14159265358979323846264338327950288419716939937510;
        long double PI2 = 3.14159265358979323846264338327950288419716939937510L;
        numa = 12;
        numa = 84;
        int num2a = 32;
        return 0;
        /* END */
}