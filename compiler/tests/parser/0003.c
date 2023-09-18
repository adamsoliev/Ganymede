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
// void (*P)(void);

float f[128];
int up[15], down[15], rows[8], x[8];
// int in[] = {10, 32, -1, 567, 3, 18, 1, -51, 789, 0};
// int x[3][4], *y[3];
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
