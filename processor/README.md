
## Design

![64-bit RISC-V Core design](./assets/RISCV_29_10_23.png)

5-stage pipelined 64-bit RISC-V core
- Supported instructions: RV32I/RV64I
- Forwarding for RAW hazards
    - MEM   -> EX
    - WB    -> EX
    - WB    -> ID (implicitly through Register File)
- Stalling for load-use hazards
- Static 'not taken' branch predictor

