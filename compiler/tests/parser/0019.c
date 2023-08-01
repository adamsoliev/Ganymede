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
