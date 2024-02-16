#define NPROC 2

// Saved registers for kernel context switches.
struct context {
        uint64 ra;
        uint64 sp;

        // callee-saved
        uint64 s0;
        uint64 s1;
        uint64 s2;
        uint64 s3;
        uint64 s4;
        uint64 s5;
        uint64 s6;
        uint64 s7;
        uint64 s8;
        uint64 s9;
        uint64 s10;
        uint64 s11;
};

enum procstate { UNUSED, RUNNABLE, RUNNING };

// Per-process state
struct proc {
        enum procstate state;         // Process state
        int pid;                      // Process ID
        uint64 kstack;         // Virtual address of kernel stack
        uint64 sz;             // Size of process memory (bytes)
        uint64 *pagetable;            // User page table
        struct trapframe *trapframe;  // Data page for trampoline.S
        struct context context;       // Swtch() here to run process
        char name[16];                // Process name (debugging)
};

struct trapframe {
        /*   0 */ uint64 kernel_satp;  // kernel page table
        /*   8 */ uint64 kernel_sp;    // top of process's kernel stack
        /*  16 */ uint64 kernel_trap;  // usertrap()
        /*  24 */ uint64 epc;          // saved user program counter
        /*  40 */ uint64 ra;
        /*  48 */ uint64 sp;
        /*  56 */ uint64 gp;
        /*  64 */ uint64 tp;
        /*  72 */ uint64 t0;
        /*  80 */ uint64 t1;
        /*  88 */ uint64 t2;
        /*  96 */ uint64 s0;
        /* 104 */ uint64 s1;
        /* 112 */ uint64 a0;
        /* 120 */ uint64 a1;
        /* 128 */ uint64 a2;
        /* 136 */ uint64 a3;
        /* 144 */ uint64 a4;
        /* 152 */ uint64 a5;
        /* 160 */ uint64 a6;
        /* 168 */ uint64 a7;
        /* 176 */ uint64 s2;
        /* 184 */ uint64 s3;
        /* 192 */ uint64 s4;
        /* 200 */ uint64 s5;
        /* 208 */ uint64 s6;
        /* 216 */ uint64 s7;
        /* 224 */ uint64 s8;
        /* 232 */ uint64 s9;
        /* 240 */ uint64 s10;
        /* 248 */ uint64 s11;
        /* 256 */ uint64 t3;
        /* 264 */ uint64 t4;
        /* 272 */ uint64 t5;
        /* 280 */ uint64 t6;
};
