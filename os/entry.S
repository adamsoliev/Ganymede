.section .text
.global _entry
_entry:

    # clear 11 and 12 bits and set them to 01 (S-mode), for mret
    csrr	a5,mstatus
    lui 	a4,0xffffe
    add	    a4,a4,2047 # ffffffffffffe7ff <end+0xffffffff7ffda20f>
    and 	a5,a5,a4
    lui 	a4,0x1
    add	    a4,a4,-2048 # 800 <_entry-0x7ffff800>
    or	    a5,a5,a4
    csrw	mstatus,a5

    # li      t0, (3 << 11)
    # csrc    mstatus, t0
    # li      t0, (1 << 11)
    # csrs    mstatus, t0

    # set mepc to C's main
    la		t0, main
    csrw	mepc, t0

    # disable paging 
    csrw	satp, zero

    # delegate all M-mode interrupts and exceptions to S-mode
    li      t0, 0xffff
    csrw    medeleg, t0
    csrw    mideleg, t0

    # enable S-mode external, timer, software interrupts
    csrr    t0, sie
    ori     t0, t0, (1 << 9) | (1 << 5) | (1 << 1)
    csrw    sie, t0

    # PMP
    li	    t0, -1
    srli 	t0, t0, 0xa
    csrw	pmpaddr0, t0
    li	    t0, 15
    csrw	pmpcfg0, t0

    # the hardware ensures that mtime increases at a constant frequency, and 
    # a timer interrupt is triggered when mtime becomes greater than mtimecmp. 
    # therefore, periodic interrupts can be requested by software by setting mtimecmp reg

    # ask CLINT for timer interrupt
    # li      t0, (0x200BFF8 + 1000000)   # 1000000 cycles; about 1/10th second in qemu 
    li      t0, 0x200BFF8               # load address of mtime
    ld      t0, 0(t0)                   # load value of mtime
    li      t1, 1000000                 # load 1000000 (cycles; about 1/10th second in qemu)
    add     t0, t0, t1                  # mtime + interval 

    li      t6, 0x2004000               # load address of mtimecmp, assuming a single core: 0x2000000 + 4000 + 8(hartID), where hartID is 0
    sd      t0, 0(t6)                   # set mtimecmp

    # prepare info in timer_scratch[] for timervec
    # la      t0, timer_scratch           # load timer_scratch's address
    # sd      t6, 24(t0)                  # store mtimecmp's address at timer_scratch[3]

    # li      t1, 1000000                 # load desired interval (in cycles) 
    # sd      t1, 32(t0)                  # store desired interval at timer_scratch[4]

    # csrw    mscratch, t0

    # set M-mode trap handler
    la      t0, timervec
    csrw    mtvec, t0

    # enable M-mode interrupts
    csrr    t0, mstatus
    ori     t0, t0, (1 << 3) 
    csrw    mstatus, t0

    # enable M-mode timer interrupts
    csrr    t0, mie
    ori     t0, t0, (1 << 7) 
    csrw    mie, t0

    # return from M-mode to lower privilige mode (S in our case)
    mret

spin:
    j spin


.globl timervec
.align 4
timervec:
        # timer_scratch[0,8,16] : register save area.
        # timer_scratch[24]     : address of CLINT's mtimecmp register.
        # timer_scratch[32]     : desired interval between interrupts.
        
        csrrw a0, mscratch, a0
        sd a1, 0(a0)
        sd a2, 8(a0)
        sd a3, 16(a0)

        # schedule the next timer interrupt
        # by adding interval to mtimecmp.
        ld a1, 24(a0) # mtimecmp
        ld a2, 32(a0) # interval
        ld a3, 0(a1)
        add a3, a3, a2
        sd a3, 0(a1)

        # arrange for a supervisor software interrupt
        # after this handler returns.
        li a1, 2
        csrw sip, a1

        ld a3, 16(a0)
        ld a2, 8(a0)
        ld a1, 0(a0)
        csrrw a0, mscratch, a0

        mret


# https://danielmangum.com/posts/risc-v-bytes-timer-interrupts/

# RISC-V PRIV 

# ---------------------------
# ecall      USER        ^
# --|--------------------|---
#   |                    |
#   v                   sret
#          SUPERVISOR
#  ecall                 ^
#   |                    |
# --|--------------------|---
#   v       MACHINE     mret
# ---------------------------

# --------------------------------------------------------------------------------------
#  MACHINE
# --------------------------------------------------------------------------------------
# mstatus
# SD MBE SBE SXL[1:0] UXL[1:0] TSR TW TVM MXR SUM MPRV XS[1:0] FS[1:0] MPP[1:0] VS[1:0] SPP MPIE UBE SPIE MIE SIE 
# 63 37  36  35-34    33-32    22  21 20  19  18  17   16-15    14-13    12-11    10-9   1    1   1   5    3   1

# mie
# MEIE SEIE MTIE STIE MSIE SSIE
# 11    9    7    5    3    1   

# mip
# MEIP SEIP MTIP STIP MSIP SSIP
# 11    9    7    5    3    1   

# medeleg
# bit positions correspond to bit positions of mcause (see mcause table in latest riscv priv spec)

# mideleg
# bit positions correspond to bit positions of mie/mip

# mstatus       - M status     
# mepc          - M pc
# mie           - M interrupt enable
# medeleg       - M exception delegation
# mideleg       - M interrupt delegation
# mtvec         - M trap-vector base address
# mscratch      - M 
# mcounteren    - M counter enable
# mtime         - M real-time counter
# mtimecmp      - M timer compare
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
#  SUPERVISOR
# --------------------------------------------------------------------------------------
# sie
# SEIE STIE SSIE        (external interrupt enable) (timer ie) (software ie)
#  9    5    1

# sip
# SEIP STIP SSIP
#  9    5    1

# sstatus   - S status
# sepc      - S pc
# sie       - S interrupt enable
# sip       - S interrupt pending
# stvec     - S trap-vector base address
# satp      - S address translation and protection
# scause    - S trap cause
# stval     - S trap value
# --------------------------------------------------------------------------------------

# pmpcfg0   - physical memory protection
# pmpaddr0  - physical memory protection
