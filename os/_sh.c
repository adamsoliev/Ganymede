int write(char);
int read();

// shell
int main() {
        write('$');
        while (1) {
                int c = read();
                if (c == '\r')
                        write('\n');
                else if (c == 0x7f)
                        write('\b');
                else
                        write(c);
        }
        return 0;
}
