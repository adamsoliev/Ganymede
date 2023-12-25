unsigned long timer_scratch[5];

int main() {
        // _entry jumps here
        // install kernel trap vector
        int a = 23;
        int b = 32;
        int c = a + b;
        return 0;
}