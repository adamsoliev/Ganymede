int x = 2;

int main(void) {
        int a = 23;
        for (a = 0; a < 23; a++) {
                if (a == 10) continue;
                if (a == 16) break;
        }

L1:
        return 23;

        if (a == 100) goto L1;
        return 0;
}
