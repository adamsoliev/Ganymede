/* NEGATIVE DECLSPECS TESTS */
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

/* ----------------- ORDERED TYPES TESTS ----------------- */
struct MyStruct {
        char a;
        int b;
};

union MyUnion {
        char a;
        int b;
};

enum MyEnum { A, B, C };

// Basic Types (T)
void a;
char b;
int c;
long long cc;
float d;
double e;
struct MyStruct f;
union MyUnion g;
enum MyEnum h;

// 0b0001TTTT
// Basic Type + Compound Type: Pointer to T
int* a;
char* b;
struct MyStruct* c;
union MyUnion* d;
enum MyEnum* e;

// 0b0010TTTT
// Basic Type + Compound Type: Array of T
int f[10];
char g[20];
struct MyStruct h[5];
union MyUnion i[5];
enum MyEnum j[5];

// 0b0011TTTT
// Basic Type + Compound Type: Function returning T
int k();
char l();
struct MyStruct m();
union MyUnion n();
enum MyEnum o();

// 0b000101TTTT
// Compound Type + Compound Type: Pointer to Pointer to T
int** p;
char** q;
struct MyStruct** r;
union MyUnion** s;
enum MyEnum** t;

// 0b000110TTTT
// Compound Type + Compound Type: Pointer to Array of T
int (*u)[10];
char (*v)[20];
struct MyStruct (*w)[5];
union MyUnion (*x)[5];
enum MyEnum (*y)[5];

// 0b001001TTTT
// Compound Type + Compound Type: Array of Pointer to T
int* z[10];
char* aa[20];
struct MyStruct* bb[5];
union MyUnion* cc[5];
enum MyEnum* dd[5];

// 0b000111TTTT
// Compound Type + Compound Type: Pointer to Function returning T
int (*ee)();
char (*ff)();
struct MyStruct (*gg)();
union MyUnion (*hh)();
enum MyEnum (*ii)();

// 0b00100111TTTT
// Compound Type + Compound Type: Array of Pointers to Function returning T
int (*jj[5])();
char (*kk[5])();
struct MyStruct (*ll[5])();
union MyUnion (*mm[5])();
enum MyEnum (*nn[9])();
/* ----------------- END ----------------- */

/* ----------------- UNORDERED ----------------- */
/* storage class specifiers */
static float pi;
static int counter;
extern double temperature;
long id;
const char* message;
volatile unsigned int flag;

/* type specifiers */
short short_var;
long long_var;
long long long_long_var;
unsigned int unsigned_int_var;
unsigned short unsigned_short_var;
unsigned long unsigned_long_var;
unsigned long long unsigned_long_long_var;
float float_var;
double double_var;
long double long_double_var;
char char_var;
unsigned char unsigned_char_var;
signed char signed_char_var;
void* void_ptr;
long long int x;

struct Shape;
struct Point {
        int x;
        int y;
};
union Color {
        int rgb;
        char name;
        char name[10];
};
struct employee {
        char name[20];
        int id;
        long class;
} temp;
struct employee student, faculty, staff;
/* anonymous struct */
struct {
        float x, y;
} complex;
struct sample {
        char c;
        float* pf;
        struct sample* next;
} x;
struct somestruct {
        /* Anonymous struct */
        struct {
                int x, y;
        } point;
        int type;
} w;
struct {
        unsigned short icon : 8;
        unsigned short color : 4;
        unsigned short underline : 1;
        unsigned short blink : 1;
} screen[25][80];

enum Days { MON, TUE, WED, THU, FRI };
enum MColor { RED, GREEN, BLUE } color_var;

typedef int Integer;
typedef struct Point Point;
typedef enum Days Weekday;

signed char c;
signed short s;
signed int i;
signed long int l;
unsigned char C;
unsigned short S;
unsigned int I;
unsigned long int L;
float f;
double d;
long double D;
int b;
const int a, *x;
int i, j, *p;
int b, *y;
volatile unsigned z;
void* p;

float f[128];
int up[15], down[15], rows[8], x[8];
char*** ptrPtrPtr;
int x[3][4], *y[3];

int in[] = {10, -32, 1, 567, 3, 18, 1, -51, 789, 0};
int out[] = {
        10,
        32,
        -1,
        567,
        3,
        18,
        1,
        -51,
        789,
        0,
};
int y[4][3] = {
        {1, 3, 5},
        {2, 4, 6},
        {3, 5, 7},
};
int y[4][3] = {1, 3, 5, 2, 4, 6, 3, 5, 7};
int z[4][3] = {{1}, {2}, {3}, {4}};
struct {
        int a[3], b;
} w[] = {{1}, 2};
struct {
        int a[3], b;
} w[] = {{1}, 2};
short q[4][3][2] = {{1}, {2, 3}, {4, 5, 6}};
short q[4][3][2] = {1, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 4, 5, 6};
short q[4][3][2] = {{
                            {1},
                    },
                    {
                            {2, 3},
                    },
                    {
                            {4, 5},
                            {6},
                    }};
int a[] = {1, 2}, b[] = {3, 4, 5};
char s[] = "abc", t[3] = "abc";
char s[] = {'a', 'b', 'c', '\0'}, t[] = {'a', 'b', 'c'};

enum { member_one, member_two };
const char* nm[] = {
        [member_two] = "member two",
        [member_one] = "member one",
};

/*
we can't do this for now because it requires typedef internals
div_t answer = { .quot = 2, .rem = -1 };
*/

struct {
        int a[3], b;
} w[] = {[0].a = {1}, [1].a[0] = 2};

// int a[MAX] = {1, 3, 5, 7, 9, [MAX - 5] = 8, 6, 4, 2, 0};
union {
        int any_member;
} u = {.any_member = 42};

int queens(), print();

int* a[10];                       // a is an array of pointer to int
int (*a)[10];                     // a is a pointer to an array of int
int* f();                         // f is a function returning a pointer to int
int (*ff)();                      // ff is a pointer to a function returning int
int (**fff)(int, int);            // fff is a pointer to a pointer to a function returning int
const int* p;                     // p is a non-const pointer to const int
int const* p;                     // same as above
int* const p;                     // p is a const pointer to non-const int
void (*fptr)(void);               /* fptr is a pointer to a function that takes in
                                             no params and returns void */
int (*fptr)(int, void (*)(void)); /* fptr is a function pointer that takes in int and
                                             pointer to the above function */
// int(*(*fptr1)(int, void (*)(void))); /* fptr1 is pointer to function (int and
//                                             pointer to 'void(*)(void)') returns pointer to int */
/* FOR THIS TEST TO WORK, WE GOT TO HAVE A FUNCTION THAT MOVES THE COMPOUND TYPE BITS 
BY 2 TO THE RIGHT SINCE THEY WRONGLY STARTED FROM THE 3RD */

int* (*fptr2)(int, void (*)(void)); /* fptr2 = fptr1 */

void (*signal(int, void (*fp)(int)))(int);

// /*
//     1. -----------------
//     In both declarators and expressions, the '[]' and '()' operators have higher precedence
//     than the '*' operator, so you need to explicitly group it with the identifier when working
//     with pointers to arrays ((*a)[N]) and pointers to functions ((*f)())

//         signal                                  -- signal
//         signal(                    )            -- is a function taking
//         signal(                    )            --   parameter unnamed
//         signal(int,                )            --     of type int
//         signal(int,        fp      )            --   parameter fp
//         signal(int,      (*fp)     )            --     is a pointer
//         signal(int,      (*fp)(   ))            --     to a function taking
//         signal(int,      (*fp)(   ))            --       parameter unnamed
//         signal(int,      (*fp)(int))            --       of type int
//         signal(int, void (*fp)(int))            --     returning void
//         (*signal(int, void (*fp)(int)))         -- returning a pointer
//         (*signal(int, void (*fp)(int)))(   )    --  to a function taking
//         (*signal(int, void (*fp)(int)))(   )    --    parameter unnamed
//         (*signal(int, void (*fp)(int)))(int)    --    of type int
//     void (*signal(int, void (*fp)(int)))(int);  --   returning void
//     -----------------

//     2. -----------------
//     C declarations are decoded from inside out using a simple rule: start from the identifier
//     and check on the right side for '[]' (array) or '()' (function) then check on the left side
//     for the type of the values (stored in the array or returned by the function), without
//     crossing the parentheses; escape from the parentheses and repeat.

//     For example:

//     void (*p)()
//     p is (nothing on the right) a pointer (on the left, don't cross the parentheses) to
//     (escape the parentheses, read the next level) a function (right) that returns nothing (left).
//     -----------------
// */

struct s {
        double i;
} f(void);
union {
        struct {
                int f1;
                struct s f2;
        } u1;
        struct {
                struct s f3;
                int f4;
        } u2;
} g;
struct s ffff(void) { return g.u1.f2; }

extern int max(int a, int b) { return a > b ? a : b; }

void g(int (*funcp)(void)) {
        /* ... */
        (*funcp)(); /* or funcp(); ... */
        funcp();
}

void g(int func(void)) {
        /* ... */
        func(); /* or (*func)(); ...*/
}

int fffff(int (*)(), double (*)[3]);
int fffff(int (*)(char*), double (*)[]);
int fffff(int (*)(char*), double (*)[3]);

int x[3][5];
struct s {
        int i;
        const int ci;
};
struct s s;
const struct s cs;
volatile struct s vs;

union {
        struct {
                int alltypes;
        } n;
        struct {
                int type;
                int intnode;
        } ni;
        struct {
                int type;
                double doublenode;
        } nf;
} u;

int* p = (int[]){2, 4};

void ffffff(void) {
        int* p;
        /*...*/
        p = (int[2]){*p};
        /*...*/
}

const float* readOnlyFloatArray = (const float[]){1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6};
const char* readOnlyCharArray = (const char[]){"/tmp/fileXXXXXX"};

struct s {
        int i;
};
int f(void) {
        struct s *p = 0, *q;
        int j = 0;
again:
        q = p, p = &((struct s){j++});
        if (j < 2) goto again;
        return p == q && q->i == 1;
}

extern void* alloc(unsigned int);
double* dp = alloc(sizeof *dp);
int ssize = sizeof array / sizeof array[0];
int fsize3(int n) {
        char b[n + 3];
        return sizeof b;
}

const void* c_vp;
void* vp;
const int* c_ip;
volatile int* v_ip;
int* ip;
const char* c_cp;

static int i = 2 || 1 / 0;
struct s {
        int n;
        double d[];
};
struct s* p = malloc(sizeof(struct s) + sizeof(double[m]));
struct {
        int n;
        double d[m];
}* p;
struct s t1 = {0};  // valid
struct s* s1;
struct s* s2;
s1 = malloc(sizeof(struct s) + 64);
s2 = malloc(sizeof(struct s) + 46);
struct {
        int n;
        double d[8];
}* s1;
struct {
        int n;
        double d[5];
}* s2;
double* dp;
dp = &(s1->d[0]);  // valid
*dp = 42;          // valid
dp = &(s2->d[0]);  // valid
enum hue { chartreuse, burgundy, claret = 20, winedark };
enum hue col, *cp;
col = claret;
cp = &col;
struct tnode {
        int count;
        struct tnode *left, *right;
};
struct tnode s, *sp;
extern const volatile int real_time_clock;
const struct s {
        int mem;
} cs = {1};
struct s ncs;  // the object ncs is modifiable
int* pi;
const int* pci;
ncs = cs;       // valid
pi = &ncs.mem;  // valid
pci = &cs.mem;  // valid
int* restrict a;
int* restrict b;
extern int c[];
void f(int n, int* restrict p, int* restrict q) {
        while (n-- > 0) *p++ = *q++;
}
void g(void) {
        extern int d[100];
        f(50, d + 50, d);  // valid
}
void h(int n, int* restrict p, int* restrict q, int* restrict r) {
        int i;
        for (i = 0; i < n; i++) p[i] = q[i] + r[i];
}
typedef struct {
        int n;
        float* restrict v;
} vector;
inline double fahr(double t) { return (9.0 * t) / 5.0 + 32.0; }
inline double cels(double t) { return (5.0 * (t - 32.0)) / 9.0; }
extern double fahr(double);
// creates an external definition
double convert(int is_fahr, double temp) {
        /* A translator may perform inline substitutions */
        return is_fahr ? cels(temp) : fahr(temp);
}

const int* ptr_to_constant;
int* const constant_ptr;
typedef int* int_ptr;
float fa[11], *afp[17];
extern int* x;
extern int y[];
extern int n;
extern int m;
// void fcompat(void) {
//         int a[n][6][m];
//         int(*p)[4][n + 1];
//         int c[n][n][6][m];
//         int(*r)[n][n][n + 1];
//         p = a;
//         // invalid: not compatible because 4 != 6
//         r = c;
//         // compatible, but defined behavior only if
//         // n == 6 and m == n+1
// }
extern int n;
int A[n];                        // invalid: file scope VM
extern int (*p2)[n];             // invalid: file scope VLA
int B[100];                      // valid: file scope but not VM
void fvla(int m, int C[m][m]);   // valid: VLA with prototype scope
void fvla(int m, int C[m][m]) {  // valid: adjusted to auto pointer to VLA
        typedef int VLA[m][m];   // valid: block scope typedef VLA
        struct tag {
                int (*y)[n];  // invalid: y not ordinary identifier
                int z[n];     // invalid: z not ordinary identifier
        };
        int D[m];                // valid: auto VLA
        static int E[m];         // invalid: static block scope VLA
        extern int F[m];         // invalid: F has linkage and is VLA
        int(*s)[m];              // valid: auto pointer to VLA
        extern int(*r)[m];       // invalid: r has linkage and points to VLA
        static int(*q)[m] = &B;  // valid: q is a static block pointer to VLA
}
int f(void), *fip(), (*pfi)();
int (*apfi[3])(int* x, int* y);

int (*fpfi(int (*)(long), int))(int, ...);
void addscalar(int n, int m, double a[n][n * m + 300], double x);
int main() {
        double b[4][308];
        addscalar(4, 2, b, 2.17);
        return 0;
}
void addscalar(int n, int m, double a[n][n * m + 300], double x) {
        for (int i = 0; i < n; i++)
                for (int j = 0, k = n * m + 300; j < k; j++)
                        // a is a pointer to a VLA with n*m+300 elements
                        a[i][j] += x;
}
double maximum(int n, int m, double a[n][m]);
double maximum(int n, int m, double a[*][*]);
double maximum(int n, int m, double a[][*]);
double maximum(int n, int m, double a[][m]);
void f(double (*restrict a)[5]);
void f(double a[restrict][5]);
void f(double a[restrict 3][5]);
void f(double a[restrict static 3][5]);

int x[] = {1, 3, 5};

// int main(void) {
//         if (*cp != burgundy) a = 23;
//         char c;
//         int i;
//         long l;
//         l = (c = i);
//         t1.n = 4;  // valid

//         f(a, (t = 3, t + 2), c);
//         if ((c = f()) == -1) {
//                 char c = '\n';
//         }
//         /* ... */
//         int n = 4, m = 3;
//         int a[n][m];
//         int(*p)[m] = a;  // p == &a[0]
//         p += 1;          // p == &a[1]
//         (*p)[2] = 99;    // a[1][2] == 99
//         n = p - a;       // n == 1

//         if ((const char[]){"abc"} == "abc") {
//                 a = 32;
//         }
//         (*pf[f1()])(f2(), f3() + f4());
//         g.u2.f3 = f();

//         u.nf.type = 1;
//         u.nf.doublenode = 3.14;
//         /* ... */
//         if (u.n.alltypes == 1) {
//                 if (sin(u.nf.doublenode) == 0.0) {
//                         return 0;
//                 }
//         }

//         unsigned long int a = 23;
//         for (a = 0; a < 23; a++) {
//                 if (a == 10)
//                         continue;
//                 else
//                         a = a * 32;
//                 if (a == 16) break;
//         }
//         for (;;) {
//                 break;
//         }
//         for (;;)
//                 ;
//         for (int i = 0; i < 10; i++) {
//                 if (i == 10) return 0;
//         }
//         while (a < 23) {
//                 a = 23 + 32;
//         }
//         do {
//                 a = 23 + 54;
//         } while (a > 10);
//         switch (a) {
//                 case 1: break;
//                 case 2: break;
//                 default: a = 23;
//         }
//         char* s;
//         while (*s++ != '\0')
//                 ;
//         while (loop1) {
//                 /* ... */
//                 while (loop2) {
//                         /* ... */
//                         if (want_out) goto end_loop1;
//                         /* ... */
//                 }
//         /* ... */
//         end_loop1:;
//         }

//         /* ... */
//         goto first_time;
//         for (;;) {
//                 // determine next operation
//                 /* ... */
//                 if (need_to_reinitialize) {
//                 // reinitialize-only code
//                 /* ... */
//                 first_time:
//                         // general initialization code
//                         /* ... */
//                         continue;
//                 }
//                 // handle other operations
//                 /* ... */
//         }

//         while (a < 23) {
//                 /* ... */
//                 continue;
//         /* ... */
//         contin:;
//         }

// L1:
//         return 23;

//         if (a == 100) goto L1;
//         return 23 + 32 - 23 + 32;
// }

// int main() {
//         /* CONSTANTS AND SYMBOL TABLE TESTS */
//         int numa = 23;
//         char* namea = "Hello World";
//         char aa = '\a';
//         char ba = '\b';
//         char ca = '\f';
//         char da = '\n';
//         char ea = '\r';
//         char fa = '\t';
//         char ga = '\v';
//         char ha = 'a';
//         char ia = '\\';
//         char ja = '\'';
//         char ka = '\"';
//         char la = '\?';
//         float nfa = 2.3f;
//         numa = 92;
//         double PI = 3.14159265358979323846264338327950288419716939937510;
//         long double PI2 = 3.14159265358979323846264338327950288419716939937510L;
//         numa = 12;
//         numa = 84;
//         int num2a = 32;
//         return 0;
//         /* END */
// }
