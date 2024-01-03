#include "kernel.h"

.section .text
.global _entry
_entry:
    la      sp, stack0

    # clear 11,12 bits and 
    li      t0, (3 << 11)
    csrc    mstatus, t0
    # set them to '01' (S-mode), for mret
    li      t0, (1 << 11)
    csrs    mstatus, t0

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

    # HW ensures that mtime increases at a constant frequency, and 
    # a timer interrupt is triggered when mtime matches mtimecmp.
    # Thus, timer interrupts can be requested by software by setting mtimecmp 

    # set mtimecmp
    li      t0, 0x200BFF8               # load address of mtime
    ld      t0, 0(t0)                   # load value of mtime
    li      t1, 1000000                 # load 1000000 (cycles; about 1/10th second in qemu)
    add     t0, t0, t1                  # mtime + interval 
    li      t6, 0x2004000               # load address of mtimecmp, assuming a single core: 0x2000000 + 4000 + 8(hartID), where hartID is 0
    sd      t1, 0(t6)                   # set mtimecmp

    # prepare info in timer_scratch[] for timervec
    la      t0, timer_scratch           # load timer_scratch's address
    sd      t6, 24(t0)                  # store mtimecmp's address at timer_scratch[3]
    li      t1, 1000000                 # load desired interval (in cycles) 
    sd      t1, 32(t0)                  # store desired interval at timer_scratch[4]
    csrw    mscratch, t0

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

.globl kernelvec
.align 4
kernelvec:
    # make room to save registers.
    addi sp, sp, -256

    # save the registers.
    sd ra, 0(sp)
    sd sp, 8(sp)
    sd gp, 16(sp)
    sd tp, 24(sp)
    sd t0, 32(sp)
    sd t1, 40(sp)
    sd t2, 48(sp)
    sd s0, 56(sp)
    sd s1, 64(sp)
    sd a0, 72(sp)
    sd a1, 80(sp)
    sd a2, 88(sp)
    sd a3, 96(sp)
    sd a4, 104(sp)
    sd a5, 112(sp)
    sd a6, 120(sp)
    sd a7, 128(sp)
    sd s2, 136(sp)
    sd s3, 144(sp)
    sd s4, 152(sp)
    sd s5, 160(sp)
    sd s6, 168(sp)
    sd s7, 176(sp)
    sd s8, 184(sp)
    sd s9, 192(sp)
    sd s10, 200(sp)
    sd s11, 208(sp)
    sd t3, 216(sp)
    sd t4, 224(sp)
    sd t5, 232(sp)
    sd t6, 240(sp)

    # call the C trap handler in trap.c
    call kerneltrap

    # restore registers.
    ld ra, 0(sp)
    ld sp, 8(sp)
    ld gp, 16(sp)
    # not tp (contains hartid), in case we moved CPUs
    ld t0, 32(sp)
    ld t1, 40(sp)
    ld t2, 48(sp)
    ld s0, 56(sp)
    ld s1, 64(sp)
    ld a0, 72(sp)
    ld a1, 80(sp)
    ld a2, 88(sp)
    ld a3, 96(sp)
    ld a4, 104(sp)
    ld a5, 112(sp)
    ld a6, 120(sp)
    ld a7, 128(sp)
    ld s2, 136(sp)
    ld s3, 144(sp)
    ld s4, 152(sp)
    ld s5, 160(sp)
    ld s6, 168(sp)
    ld s7, 176(sp)
    ld s8, 184(sp)
    ld s9, 192(sp)
    ld s10, 200(sp)
    ld s11, 208(sp)
    ld t3, 216(sp)
    ld t4, 224(sp)
    ld t5, 232(sp)
    ld t6, 240(sp)

    addi sp, sp, 256

    sret

.globl swtch
swtch:
    sd ra, 0(a0)
    sd sp, 8(a0)
    sd s0, 16(a0)
    sd s1, 24(a0)
    sd s2, 32(a0)
    sd s3, 40(a0)
    sd s4, 48(a0)
    sd s5, 56(a0)
    sd s6, 64(a0)
    sd s7, 72(a0)
    sd s8, 80(a0)
    sd s9, 88(a0)
    sd s10, 96(a0)
    sd s11, 104(a0)

    ld ra, 0(a1)
    ld sp, 8(a1)
    ld s0, 16(a1)
    ld s1, 24(a1)
    ld s2, 32(a1)
    ld s3, 40(a1)
    ld s4, 48(a1)
    ld s5, 56(a1)
    ld s6, 64(a1)
    ld s7, 72(a1)
    ld s8, 80(a1)
    ld s9, 88(a1)
    ld s10, 96(a1)
    ld s11, 104(a1)
    
    ret


.align 4
.globl uservec
uservec:    
    # trap.c sets stvec to point here, so
    # traps from user space start here,
    # in supervisor mode

    # save user a0 in sscratch so
    # a0 can be used to get at TRAPFRAME.
    csrw sscratch, a0

    # each process has a separate p->trapframe memory area
    li a0, TRAPFRAME
    
    # save the user registers in TRAPFRAME
    sd ra, 40(a0)
    sd sp, 48(a0)
    sd gp, 56(a0)
    sd tp, 64(a0)
    sd t0, 72(a0)
    sd t1, 80(a0)
    sd t2, 88(a0)
    sd s0, 96(a0)
    sd s1, 104(a0)
    sd a1, 120(a0)
    sd a2, 128(a0)
    sd a3, 136(a0)
    sd a4, 144(a0)
    sd a5, 152(a0)
    sd a6, 160(a0)
    sd a7, 168(a0)
    sd s2, 176(a0)
    sd s3, 184(a0)
    sd s4, 192(a0)
    sd s5, 200(a0)
    sd s6, 208(a0)
    sd s7, 216(a0)
    sd s8, 224(a0)
    sd s9, 232(a0)
    sd s10, 240(a0)
    sd s11, 248(a0)
    sd t3, 256(a0)
    sd t4, 264(a0)
    sd t5, 272(a0)
    sd t6, 280(a0)

    # save the user a0 in p->trapframe->a0
    csrr t0, sscratch
    sd t0, 112(a0)
    # initialize kernel stack pointer, from p->trapframe->kernel_sp
    ld sp, 8(a0)
    # make tp hold the current hartid, from p->trapframe->kernel_hartid
    ld tp, 32(a0)
    # load the address of usertrap(), from p->trapframe->kernel_trap
    ld t0, 16(a0)
    # fetch the kernel page table address, from p->trapframe->kernel_satp.
    ld t1, 0(a0)
    # jump to usertrap(), which does not return
    jr t0

.globl userret
.align 4
userret:
    # userret()
    # called by usertrapret() in trap.c to
    # switch from kernel to user.

    # li a0, TRAPFRAME

    # restore all but a0 from TRAPFRAME
    ld ra, 40(a0)
    ld sp, 48(a0)
    ld gp, 56(a0)
    ld tp, 64(a0)
    ld t0, 72(a0)
    ld t1, 80(a0)
    ld t2, 88(a0)
    ld s0, 96(a0)
    ld s1, 104(a0)
    ld a1, 120(a0)
    ld a2, 128(a0)
    ld a3, 136(a0)
    ld a4, 144(a0)
    ld a5, 152(a0)
    ld a6, 160(a0)
    ld a7, 168(a0)
    ld s2, 176(a0)
    ld s3, 184(a0)
    ld s4, 192(a0)
    ld s5, 200(a0)
    ld s6, 208(a0)
    ld s7, 216(a0)
    ld s8, 224(a0)
    ld s9, 232(a0)
    ld s10, 240(a0)
    ld s11, 248(a0)
    ld t3, 256(a0)
    ld t4, 264(a0)
    ld t5, 272(a0)
    ld t6, 280(a0)

    # restore user a0
    ld a0, 112(a0)
    
    # return to user mode and user pc.
    # usertrapret() set up sstatus and sepc.
    sret


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
