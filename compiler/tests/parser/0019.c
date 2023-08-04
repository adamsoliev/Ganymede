int func() { return 2; }

int func1(int a) { return a; }

// int func2(int a, int b) {
//     return a + b;
// };

// int func3(int a, int b, int c) {
//     return a + b + c;
// };

// int func4(int a, int b, int c, int d) {
//     return a + b + c + d;
// };

// int func5(int a, int b, int c, int d, int e) {
//     return a + b + c + d + e;
// };

// int func6(int a, int b, int c, int d, int e, int f) {
//     return a + b + c + d + e + f;
// };

// int func7(int a, int b, int c, int d, int e, int f, int g) {
//     return a + b + c + d + e + f + g;
// };

int main() {
    int a = func();
    if (a == 0) {
        return func();
    }
    int b = func1(a);
    if (b == 0) {
        return func1(a);
    }
    int c = func2(a, b);
    if (c == 0) {
        return func2(a, b);
    }
    int d = func3(a, b, c);
    if (d == 0) {
        return func3(a, b, c);
    }
    int e = func4(a, b, c, d);
    if (e == 0) {
        return func4(a, b, c, d);
    }
    int f = func5(a, b, c, d, e);
    if (f == 0) {
        return func5(a, b, c, d, e);
    }
    int g = func6(a, b, c, d, e, f);
    if (g == 0) {
        return func6(a, b, c, d, e, f);
    }
    int h = func7(a, b, c, d, e, f, g);
    if (h == 0) {
        return func7(a, b, c, d, e, f, g);
    }
    return h;
}
