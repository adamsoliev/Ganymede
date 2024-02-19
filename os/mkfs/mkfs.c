#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "kernel/types.h"

#define DIRNAMESZ 16
#define NDIRECT 12

#define BSIZE 4096

#define FSSIZE 64
#define NIBLOCKS 4
#define NDBLOCKS 56

struct superblock {
        uint magic;
        uint size;
        uint ninodes;     // number of inode blocks
        uint ndblocks;    // number of data blocks
        uint inodestart;  // first inode block number
};

struct inode {
        uint inum;  // inode number
        uint type;  // file or dir
        uint size;
        uint64 addrs[NDIRECT];
};

struct dirent {
        uint inum;
        char name[DIRNAMESZ];
};

/*
Disk layout:
[ boot | sb | inodebitmap | dbitmap  | inodes | data ]
[   1  |  1 |      1      |     1    |   4    |  56  ] => 64 blocks in total, each is 4KB (256 KB)

*/

int fd;
struct superblock sb;
uint freeblock;
char zeroes[BSIZE];

// convert to riscv byte order
uint xint(uint x) {
        uint y;
        uchar *a = (uchar *)&y;
        a[0] = x;
        a[1] = x >> 8;
        a[2] = x >> 16;
        a[3] = x >> 24;
        return y;
}

void wsect(uint sec, void *buf);
void die(const char *s);

int main(int argc, char *argv[]) {
        if (argc < 2) {
                printf("usage: mkfs fs.img files...\n");
                exit(1);
        }
        fd = open(argv[1], O_RDWR | O_CREAT | O_TRUNC, 0666);
        if (fd < 0) exit(1);

        sb.magic = 0x10203040;
        sb.size = xint(FSSIZE);
        sb.ninodes = NIBLOCKS;
        sb.ndblocks = NDBLOCKS;
        sb.inodestart = 4;

        freeblock = 4 + NIBLOCKS;

        for (int i = 0; i < FSSIZE; i++) {
                wsect(i, zeroes);
        }

        char buf[BSIZE];
        memset(buf, 0, sizeof(buf));
        memmove(buf, &sb, sizeof(sb));
        wsect(1, buf);
}

void wsect(uint sec, void *buf) {
        if (lseek(fd, sec * BSIZE, 0) != sec * BSIZE) die("lseek");
        if (write(fd, buf, BSIZE) != BSIZE) die("write");
}

void die(const char *s) {
        perror(s);
        exit(1);
}
