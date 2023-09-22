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
int z[4][3] = {{1}, {2}, {3}, {4}};
struct {
        int a[3], b;
} w[] = {{1}, 2};
short q[4][3][2] = {{1}, {2, 3}, {4, 5, 6}};
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

int a[MAX] = {1, 3, 5, 7, 9, [MAX - 5] = 8, 6, 4, 2, 0};
union {
        int any_member;
} u = {.any_member = 42};

int queens(), print();

int* a[10];                               // a is an array of pointer to int
int (*a)[10];                             // a is a pointer to an array of int
int* f();                                 // f is a function returning a pointer to int
int (*f)();                               // f is a pointer to a function returning int
const int* p;                             // p is a non-const pointer to const int
int const* p;                             // same as above
int* const p;                             // p is a const pointer to non-const int
void (*fptr)(void);                       /* fptr is a pointer to a function that takes in
                                             no params and returns void */
int (*fptr)(int, void (*)(void));         /* fptr is a function pointer that takes in int and
//                                              pointer to the above function */
int(*(*ptr_to_ptr)(int, void (*)(void))); /* ptr_to_ptr is a pointer to a pointer
//                                              that points to the above function */
void (*signal(int, void (*fp)(int)))(int);

/*
    1. -----------------
    In both declarators and expressions, the '[]' and '()' operators have higher precedence
    than the '*' operator, so you need to explicitly group it with the identifier when working
    with pointers to arrays ((*a)[N]) and pointers to functions ((*f)())

        signal                                  -- signal
        signal(                    )            -- is a function taking
        signal(                    )            --   parameter unnamed
        signal(int,                )            --     of type int
        signal(int,        fp      )            --   parameter fp
        signal(int,      (*fp)     )            --     is a pointer
        signal(int,      (*fp)(   ))            --     to a function taking
        signal(int,      (*fp)(   ))            --       parameter unnamed
        signal(int,      (*fp)(int))            --       of type int
        signal(int, void (*fp)(int))            --     returning void
        (*signal(int, void (*fp)(int)))         -- returning a pointer
        (*signal(int, void (*fp)(int)))(   )    --  to a function taking
        (*signal(int, void (*fp)(int)))(   )    --    parameter unnamed
        (*signal(int, void (*fp)(int)))(int)    --    of type int
    void (*signal(int, void (*fp)(int)))(int);  --   returning void
    -----------------

    2. -----------------
    C declarations are decoded from inside out using a simple rule: start from the identifier
    and check on the right side for '[]' (array) or '()' (function) then check on the left side
    for the type of the values (stored in the array or returned by the function), without
    crossing the parentheses; escape from the parentheses and repeat.

    For example:

    void (*p)()
    p is (nothing on the right) a pointer (on the left, don't cross the parentheses) to
    (escape the parentheses, read the next level) a function (right) that returns nothing (left).
    -----------------
*/

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
struct s f(void) { return g.u1.f2; }
/* ... */

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

int f(int (*)(), double (*)[3]);
int f(int (*)(char*), double (*)[]);
int f(int (*)(char*), double (*)[3]);

i = i + 1;
a[i] = i;
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

int main(void) {
        (*pf[f1()])(f2(), f3() + f4());
        g.u2.f3 = f();

        u.nf.type = 1;
        u.nf.doublenode = 3.14;
        /* ... */
        if (u.n.alltypes == 1) {
                if (sin(u.nf.doublenode) == 0.0) {
                        return 0;
                }
        }

        unsigned long int a = 23;
        for (a = 0; a < 23; a++) {
                if (a == 10)
                        continue;
                else
                        a = a * 32;
                if (a == 16) break;
        }
        for (;;) {
                break;
        }
        for (;;)
                ;
        for (int i = 0; i < 10; i++) {
                if (i == 10) return 0;
        }
        while (a < 23) {
                a = 23 + 32;
        }
        do {
                a = 23 + 54;
        } while (a > 10);
        switch (a) {
                case 1: break;
                case 2: break;
                default: a = 23;
        }
        char* s;
        while (*s++ != '\0')
                ;
        while (loop1) {
                /* ... */
                while (loop2) {
                        /* ... */
                        if (want_out) goto end_loop1;
                        /* ... */
                }
        /* ... */
        end_loop1:;
        }

        /* ... */
        goto first_time;
        for (;;) {
                // determine next operation
                /* ... */
                if (need_to_reinitialize) {
                // reinitialize-only code
                /* ... */
                first_time:
                        // general initialization code
                        /* ... */
                        continue;
                }
                // handle other operations
                /* ... */
        }

        while (a < 23) {
                /* ... */
                continue;
        /* ... */
        contin:;
        }

L1:
        return 23;

        if (a == 100) goto L1;
        return 23 + 32 - 23 + 32;
}
