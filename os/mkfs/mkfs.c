#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include "kernel/types.h"

#define DIRNAMESZ 16
#define NDIRECT 12

#define BSIZE 4096

#define FSSIZE 64
#define NIBLOCKS 5
#define NDBLOCKS 56

// Inodes per block.
#define IPB (BSIZE / sizeof(struct inode))

// Block containing inode i
#define IBLOCK(i, sb) (((i) / IPB) + sb.inodestart)

struct superblock {
        uint magic;
        uint size;
        uint ninodes;     // number of inode blocks
        uint ndblocks;    // number of data blocks
        uint inodestart;  // first inode block number
};

enum inode_type { TDIR = 1, TFILE = 2 };

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
[ boot | sb | bitmap | inodes | data ]
[   1  |  1 |   1    |   5    |  56  ] => 64 blocks in total, each is 4KB (256 KB)
    *bitmap can technically support 4096 * 8 data blocks (128 MB)
*/

int fd;
struct superblock sb;
uint freeblock;
uint freeinode = 1;
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

void rsect(uint sec, void *buf);
void wsect(uint sec, void *buf);
void rinode(uint inum, struct inode *in);
void winode(uint inum, struct inode *in);
uint ialloc(uint type);
void balloc(int used);
void iappend(uint inum, void *xp, int n);
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
        sb.inodestart = 3;

        freeblock = 3 + NIBLOCKS;

        for (int i = 0; i < FSSIZE; i++) {
                wsect(i, zeroes);
        }

        char buf[BSIZE];
        memset(buf, 0, sizeof(buf));
        memmove(buf, &sb, sizeof(sb));
        wsect(1, buf);

        uint rootino = ialloc(TDIR);
        assert(rootino == 1);

        struct dirent de;
        bzero(&de, sizeof(de));
        de.inum = xint(rootino);
        strcpy(de.name, ".");
        iappend(rootino, &de, sizeof(de));

        bzero(&de, sizeof(de));
        de.inum = xint(rootino);
        strcpy(de.name, "..");
        iappend(rootino, &de, sizeof(de));

        int i, cc, fd1;
        for (i = 2; i < argc; i++) {
                char *name;
                if (strncmp(argv[i], "user/", 5) == 0) {
                        name = argv[i] + 5;
                } else {
                        name = argv[i];
                }
                assert(index(name, '/') == 0);
                if (name[0] == '_') name += 1;

                uint inum = ialloc(TFILE);

                bzero(&de, sizeof(de));
                de.inum = xint(inum);
                strncpy(de.name, name, DIRNAMESZ);
                iappend(rootino, &de, sizeof(de));

                if ((fd1 = open(argv[i], 0)) < 0) die(argv[i]);
                while ((cc = read(fd1, buf, sizeof(buf))) > 0) iappend(inum, buf, cc);
                close(fd1);
        }

        struct inode in;
        rinode(rootino, &in);
        uint off = xint(in.size);
        off = ((off / BSIZE) + 1) * BSIZE;
        in.size = xint(off);
        winode(rootino, &in);

        balloc(freeblock);
        exit(0);
}

void rsect(uint sec, void *buf) {
        if (lseek(fd, sec * BSIZE, 0) != sec * BSIZE) die("lseek");
        if (read(fd, buf, BSIZE) != BSIZE) die("read");
}

void wsect(uint sec, void *buf) {
        if (lseek(fd, sec * BSIZE, 0) != sec * BSIZE) die("lseek");
        if (write(fd, buf, BSIZE) != BSIZE) die("write");
}

void winode(uint inum, struct inode *in) {
        char buf[BSIZE];
        uint bn;
        struct inode *ip;

        bn = IBLOCK(inum, sb);
        rsect(bn, buf);
        ip = ((struct inode *)buf) + (inum % IPB);
        *ip = *in;
        wsect(bn, buf);
}

void rinode(uint inum, struct inode *in) {
        char buf[BSIZE];
        uint bn;
        struct inode *ip;

        bn = IBLOCK(inum, sb);
        rsect(bn, buf);
        ip = ((struct inode *)buf) + (inum % IPB);
        *in = *ip;
}

uint ialloc(uint type) {
        uint inum = freeinode++;
        struct inode in;

        bzero(&in, sizeof(in));
        in.type = xint(type);
        in.size = xint(0);
        winode(inum, &in);
        return inum;
}

void balloc(int used) {
        uchar buf[BSIZE];

        printf("balloc: first %d blocks have been allocated\n", used);
        assert(used < BSIZE * 8);
        bzero(buf, BSIZE);
        for (int i = 0; i < used; i++) {
                buf[i / 8] = buf[i / 8] | (0x1 << (i % 8));
        }
        printf("balloc: write bitmap block at sector %d\n", 2);
        wsect(2, buf);
}

#define min(a, b) ((a) < (b) ? (a) : (b))

void iappend(uint inum, void *xp, int n) {
        struct inode in;
        rinode(inum, &in);
        uint off = xint(in.size);
        while (n > 0) {
                uint fbn = off / BSIZE;
                assert(fbn < NDIRECT);
                if (xint(in.addrs[fbn]) == 0) {
                        in.addrs[fbn] = xint(freeblock++);
                }
                uint x = xint(in.addrs[fbn]);

                char *p = (char *)xp;
                char buf[BSIZE];
                uint n1 = min(n, (fbn + 1) * BSIZE - off);
                rsect(x, buf);
                bcopy(p, buf + off - (fbn * BSIZE), n1);
                wsect(x, buf);
                n -= n1;
                off += n1;
                p += n1;
        }
        in.size = xint(off);
        winode(inum, &in);
}

void die(const char *s) {
        perror(s);
        exit(1);
}
