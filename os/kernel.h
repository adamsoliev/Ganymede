
#define KERNBASE 0x80000000L
#define PHYSTOP (KERNBASE + 128 * 1024 * 1024)

#define PGSIZE 4096

#define NPROC 4

#define TRAMPOLINE (PHYSTOP - PGSIZE)
// #define KSTACK(p) (TRAMPOLINE - ((p) + 1) * 2 * PGSIZE)

#define TRAPFRAME (TRAMPOLINE - PGSIZE)

#define PGROUNDUP(sz) (((sz) + PGSIZE - 1) & ~(PGSIZE - 1))
#define PGROUNDDOWN(a) (((a)) & ~(PGSIZE - 1))

//////////////////////
// FS
//////////////////////
#define ROOTINO 1
#define BSIZE 1024

// Disk layout:
// [ boot block | super block | log | inode blocks |
//                                          free bit map | data blocks]
//
// mkfs computes the super block and builds an initial file system. The
// super block describes the disk layout:
struct superblock {
        unsigned int magic;       // Must be FSMAGIC
        unsigned int size;        // Size of file system image (blocks)
        unsigned int nblocks;     // Number of data blocks
        unsigned int ninodes;     // Number of inodes.
        unsigned int nlog;        // Number of log blocks
        unsigned int logstart;    // Block number of first log block
        unsigned int inodestart;  // Block number of first inode block
        unsigned int bmapstart;   // Block number of first free map block
};

#define FSMAGIC 0x10203040

#define NDIRECT 12
#define NINDIRECT (BSIZE / sizeof(unsigned int))
#define MAXFILE (NDIRECT + NINDIRECT)

// On-disk inode structure
struct dinode {
        short type;                       // File type
        short major;                      // Major device number (T_DEVICE only)
        short minor;                      // Minor device number (T_DEVICE only)
        short nlink;                      // Number of links to inode in file system
        unsigned int size;                // Size of file (bytes)
        unsigned int addrs[NDIRECT + 1];  // Data block addresses
};

// Inodes per block.
#define IPB (BSIZE / sizeof(struct dinode))

// Block containing inode i
#define IBLOCK(i, sb) ((i) / IPB + sb.inodestart)

// Bitmap bits per block
#define BPB (BSIZE * 8)

// Block of free map containing bit for block b
#define BBLOCK(b, sb) ((b) / BPB + sb.bmapstart)

// Directory is a file containing a sequence of dirent structures.
#define DIRSIZ 14

struct dirent {
        unsigned short inum;
        char name[DIRSIZ];
};

//////////////////////
// STAT
//////////////////////
#define T_DIR 1     // Directory
#define T_FILE 2    // File
#define T_DEVICE 3  // Device

struct stat {
        int dev;             // File system's disk device
        unsigned int ino;    // Inode number
        short type;          // Type of file
        short nlink;         // Number of links to file
        unsigned long size;  // Size of file in bytes
};
