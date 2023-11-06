`default_nettype none

module Memory (
    input             clk,
    input      [63:0] mem_addr,  // instr address to be read
    output reg [31:0] mem_rdata, // instr read from memory
    input   	      mem_rstrb, // goes high when processor wants to read instr

    input [63:0]      mem_daddr, // data address
    output [63:0]     mem_drdata, // data data
    input             mem_drstrb // data strobe
);
    reg [31:0] IMEM [0:4096];
    reg [63:0] DMEM [0:50];
    initial begin 
        $readmemh("./test_a/mem_instr", IMEM);
        $readmemh("./test_a/mem_data", DMEM);
    end

    always @(posedge clk) begin
        if(mem_rstrb) begin
            mem_rdata <= IMEM[mem_addr[63:2]];
        end 
        if(mem_drstrb) begin
            mem_drdata <= DMEM[(mem_daddr[63:0] - {{48{1'b0}}, 16'h2000})/8];
            $display("DATA ADDRESS: %0d", (mem_daddr[63:0] - {{48{1'b0}}, 16'h2000})/8);
            $display("MEMORY   : %h", DMEM[0]);
            $display("MEMORY   : %h", DMEM[1]);
            $display("MEMORY   : %h", DMEM[2]);
            $display("MEMORY   : %h", DMEM[3]);
        end 
        else begin
            mem_drdata <= 0;
        end
    end
endmodule

module Processor (
    input clk,
    input reset,

    output [63:0]   mem_addr,  // instr address
    input [31:0]    mem_rdata, // instr 
    output 	        mem_rstrb, // instr strobe

    output [63:0]   mem_daddr, // data address
    input [63:0]    mem_drdata, // data data
    output          mem_drstrb // data strobe
);

    ////////////////////////////////////////////////////////////////////////////////
    // FETCH
    ////////////////////////////////////////////////////////////////////////////////
    reg [63:0] PC;
    reg [31:0] instruction;

    ////////////////////////////////////////////////////////////////////////////////
    // DECODE
    ////////////////////////////////////////////////////////////////////////////////
    // 10 RISC-V instructions
    wire isLOAD         = (instruction[6:0] == 7'b0000011); // rd <- mem[rs1+Iimm]
    wire isLOAD_FP      = (instruction[6:0] == 7'b0000111);
    wire isMISC_MEM     = (instruction[6:0] == 7'b0001111);
    wire isOP_IMM       = (instruction[6:0] == 7'b0010011); // rd <- rs1 OP Iimm
    wire isAUIPC        = (instruction[6:0] == 7'b0010111); // rd <- PC + Uimm
    wire isOP_IMM_32    = (instruction[6:0] == 7'b0011011);
    wire isSTORE        = (instruction[6:0] == 7'b0100011); // mem[rs1+Simm] <- rs2
    wire isSTORE_FP     = (instruction[6:0] == 7'b0100111);
    wire isAMO          = (instruction[6:0] == 7'b0101111);
    wire isOP           = (instruction[6:0] == 7'b0110011); // rd <- rs1 OP rs2   
    wire isLUI          = (instruction[6:0] == 7'b0110111); // rd <- Uimm   
    wire isOP_32        = (instruction[6:0] == 7'b0111011);
    wire isMADD         = (instruction[6:0] == 7'b1000011);
    wire isMSUB         = (instruction[6:0] == 7'b1000111);
    wire isNMSUB        = (instruction[6:0] == 7'b1001011);
    wire isNMADD        = (instruction[6:0] == 7'b1001111);
    wire isOP_FP        = (instruction[6:0] == 7'b1010011);
    wire isBRANCH       = (instruction[6:0] == 7'b1100011); // if(rs1 OP rs2) PC<-PC+Bimm
    wire isJALR         = (instruction[6:0] == 7'b1100111); // rd <- PC+4; PC<-rs1+Iimm
    wire isJAL          = (instruction[6:0] == 7'b1101111); // rd <- PC+4; PC<-PC+Jimm
    wire isSYSTEM       = (instruction[6:0] == 7'b1110011); // special
    // wire [6:0] opcode = instruction[6:0];

    // 5 immediate formats
    wire [31:0] Uimm = { instruction[31:12], 12'b0 };
    wire [31:0] Iimm = { {20{instruction[31]}}, instruction[31:20] };
    // wire [31:0] Simm = { {20{instruction[31]}}, instruction[31:25], instruction[11:7] };
    wire [31:0] Bimm = { {20{instruction[31]}}, instruction[7], instruction[30:25], instruction[11:8], 1'b0 };
    wire [31:0] Jimm = { {11{instruction[31]}}, instruction[31], instruction[19:12], instruction[20], instruction[30:21], 1'b0 };

    // source and destination registers
    wire [4:0] rs1Id = instruction[19:15];
    wire [4:0] rs2Id = instruction[24:20];
    wire [4:0] rdId  = instruction[11:7];

    // function codes
    wire [2:0] funct3 = instruction[14:12];
    wire [6:0] funct7 = instruction[31:25];

    ////////////////////////////////////////////////////////////////////////////////
    // EXECUTE
    ////////////////////////////////////////////////////////////////////////////////
    reg [63:0] RegisterBank [0:31];
    reg [63:0] rs1;
    reg [63:0] rs2;
    wire [63:0] writeBackData;
    wire        writeBackEn;
    
    integer i;
    initial begin
        for (i = 0; i < 32; ++i) begin
            RegisterBank[i] = 0;
        end
    end

    wire [63:0] aluIn1 = (isOP_32 || isOP_IMM_32)   ? {{32{rs1[31]}}, rs1[31:0]} : rs1;
    wire [63:0] aluIn2 = isOP_32                    ? {{32{rs2[31]}}, rs2[31:0]} : 
                         isOP                       ? rs2                        : 
                                                      {{32{Iimm[31]}}, Iimm};
    reg [63:0] aluOut = 0;
    reg [31:0] aluOutLower = 0;
    /*
    SLLI, SRLI, SRAI        => shamt[5:0] of Imm    isOP_IMM
    SLLIW, SRLIW, SRAIW     => shamt[4:0] of Imm    isOP_IMM_32

    SLL, SRL, SRA           => shamt[5:0] of rs2    isOP
    SLLW, SRLW, and SRAW    => shamt[4:0] of rs2    isOP_32
    */
    wire [5:0] shamt = (isOP)               ? rs2[5:0]          :
                       (isOP_32)            ? {1'b0, rs2[4:0]}  :
                       (isOP_IMM)           ? Iimm[5:0]         :
                                              {1'b0, Iimm[4:0]};

    always @(*) begin
        case(funct3)
            3'b000: begin
                aluOut = (funct7[5] & instruction[5]) ?  (aluIn1 - aluIn2) : (aluIn1 + aluIn2);
                if (isOP_32 || isOP_IMM_32) begin
                    aluOut = {{32{aluOut[31]}}, aluOut[31:0]};
                end
            end
            3'b001: begin
                aluOut = aluIn1 << shamt;
                if (isOP_32 || isOP_IMM_32) begin
                    aluOut = {{32{aluOut[31]}}, aluOut[31:0]};
                end
            end
            3'b010: aluOut = {63'b0, ($signed(aluIn1) < $signed(aluIn2))};
            3'b011: aluOut = {63'b0, (aluIn1 < aluIn2)};
            3'b100: aluOut = (aluIn1 ^ aluIn2);
            3'b101: 
                if (isOP_IMM || isOP_IMM_32) begin
                    if (isOP_IMM_32) begin
                        aluOutLower = Iimm[10] ? ($signed(aluIn1[31:0]) >>> shamt) : ($signed(aluIn1[31:0]) >> shamt); 
                        aluOut = {{32{aluOutLower[31]}}, aluOutLower[31:0]};
                    end
                    else begin
                        aluOut = Iimm[10] ? ($signed(aluIn1) >>> shamt) : ($signed(aluIn1) >> shamt); 
                    end
                end
                else begin
                    if (isOP_32) begin
                        aluOutLower = funct7[5] ? ($signed(aluIn1[31:0]) >>> shamt) : ($signed(aluIn1[31:0]) >> shamt); 
                        aluOut = {{32{aluOutLower[31]}}, aluOutLower[31:0]};
                    end
                    else begin
                        aluOut = funct7[5] ? ($signed(aluIn1) >>> shamt) : ($signed(aluIn1) >> shamt); 
                    end
                end
            3'b110: aluOut = (aluIn1 | aluIn2);
            3'b111: aluOut = (aluIn1 & aluIn2);	
        endcase
    end

    // The predicate for branch instructions
    reg takeBranch;
    always @(*) begin
        case(funct3)
            3'b000: takeBranch = (rs1 == rs2);
            3'b001: takeBranch = (rs1 != rs2);
            3'b100: takeBranch = ($signed(rs1) < $signed(rs2));
            3'b101: takeBranch = ($signed(rs1) >= $signed(rs2));
            3'b110: takeBranch = (rs1 < rs2);
            3'b111: takeBranch = (rs1 >= rs2);
            default: takeBranch = 1'b0;
        endcase
    end

    ////////////////////////////////////////////////////////////////////////////////
    // MEMORY ACCESS
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // WRITE BACK
    ////////////////////////////////////////////////////////////////////////////////
    assign writeBackData = (isJAL || isJALR) ? (PC + 4) :
                                (isLUI)      ? {{32{Uimm[31]}}, Uimm} :
                                (isAUIPC)    ? (PC + {{32{Uimm[31]}}, Uimm}) : 
                                isLOAD       ? LOAD_data :
                                               aluOut;
    
    assign writeBackEn = (state == EXECUTE && 
                (isOP || isOP_32 || isOP_IMM_32 ||
                isOP_IMM || 
                isJAL    || 
                isJALR   ||
                isLUI    ||
                isAUIPC)
                ) || (state == WAIT_DATA);
    // next PC
    wire [63:0] nextPC = (isBRANCH && takeBranch) ? PC  + {{32{Bimm[31]}}, Bimm}  :	       
                        isJAL                     ? PC  + {{32{Jimm[31]}}, Jimm}  :
                        isJALR                    ? rs1 + {{32{Iimm[31]}}, Iimm}  :
                        PC+4;

    /*
    op       funct3  
    -----------------------
    0000011  011        ld
    0000011  010        lw
    0000011  110        lwu
    0000011  001        lh
    0000011  101        lhu
    0000011  000        lb
    0000011  100        lbu

    0100011  011        sd
    0100011  010        sw
    0100011  001        sh
    0100011  000        sb
    */
    // LOAD
    wire [63:0] loadstore_addr = rs1 + {{32{Iimm[31]}}, Iimm};
    wire [31:0] LOAD_word = loadstore_addr[2] ? mem_drdata[63:32] : mem_drdata[31:0];
    wire [15:0] LOAD_halfword = loadstore_addr[1] ? LOAD_word[31:16] : LOAD_word[15:0];
    wire [7:0] LOAD_byte = loadstore_addr[0] ? LOAD_halfword[15:8] : LOAD_halfword[7:0];

    wire mem_byteAccess     = funct3[1:0] == 2'b00;
    wire mem_halfwordAccess = funct3[1:0] == 2'b01;
    wire mem_wordAccess     = funct3[1:0] == 2'b10;

    wire LOAD_sign = !funct3[2] & (mem_byteAccess       ? LOAD_byte[7] : 
                                   mem_halfwordAccess   ? LOAD_halfword[15] :
                                                          LOAD_word[31]);
    
    reg [63:0] LOAD_data = mem_byteAccess      ? {{56{LOAD_sign}}, LOAD_byte}     :
                           mem_halfwordAccess  ? {{48{LOAD_sign}}, LOAD_halfword} :
                           mem_wordAccess      ? {{32{LOAD_sign}}, LOAD_word}     :
                                                 mem_drdata;

    // The state machine
    localparam FETCH_INSTR = 0;
    localparam WAIT_INSTR  = 1;
    localparam FETCH_REGS  = 2;
    localparam EXECUTE     = 3;
    localparam LOAD        = 4;
    localparam WAIT_DATA   = 5;
    reg [2:0] state = FETCH_INSTR;

    always @(posedge clk) begin
        if (!reset) begin
            PC    <= 64'h00000000;
            state <= FETCH_INSTR;
        end else begin
            if (writeBackEn && rdId != 0) begin
                RegisterBank[rdId] <= writeBackData;
                // $display("state: %d", state);
                // $display("LOAD_data: %h", LOAD_data);
                // $display("writeBackData: %h", writeBackData);
                // $display("mem_byteAccess    : %h", mem_byteAccess    );
                // $display("mem_halfwordAccess: %h", mem_halfwordAccess);
                // $display("mem_wordAccess    : %h", mem_wordAccess    );
            end
            case(state)
                FETCH_INSTR: begin
                    state <= WAIT_INSTR;
                end
                WAIT_INSTR: begin
                    instruction <= mem_rdata;
                    state <= FETCH_REGS;
                end
                FETCH_REGS: begin
                    rs1 <= RegisterBank[rs1Id];
                    rs2 <= RegisterBank[rs2Id];
                    state <= EXECUTE;
                end
                EXECUTE: begin
                    PC <= nextPC;
                    state <= isLOAD ? LOAD : FETCH_INSTR;	      
                end
                LOAD: begin
                    state <= WAIT_DATA;
                end
                WAIT_DATA: begin
                    // LOAD_data <= mem_drdata;
                    state <= FETCH_INSTR;
                end
            endcase 
        end
    end

    // instruction memory
    assign mem_addr = PC;
    assign mem_rstrb = (state == FETCH_INSTR);

    // data memory
    assign mem_daddr = loadstore_addr;
    assign mem_drstrb = (state == LOAD);

    ////////////////////////////////////////////////////////////////////////////////
    // DEBUG
    ////////////////////////////////////////////////////////////////////////////////
    wire trap;
    assign trap = (isSYSTEM && ~(|Iimm) && RegisterBank[3] > 1);
    always @(posedge clk) begin
        if (trap) begin
            $display("TRAP: %0d", RegisterBank[3][31:0]);
            $finish();
        end
    end

    always @(posedge clk) begin
        if (isLOAD     ) $display("STATE: %0d   PC:%3h %h  LOAD      mem_drdata:%3h  mem_drstrb:%3h  mem_daddr:%3h  writeBackEn:%h  LOAD_data:%3h  writeBackData:%3h", state, PC, instruction, mem_drdata, mem_drstrb, mem_daddr, writeBackEn, LOAD_data, writeBackData);
        if (isLOAD_FP  ) $display("STATE: %0d   PC:%3h %h  LOAD_FP                      ", state, PC, instruction);
        if (isMISC_MEM ) $display("STATE: %0d   PC:%3h %h  MISC_MEM                     ", state, PC, instruction);
        if (isOP_IMM   ) $display("STATE: %0d   PC:%3h %h  OP_IMM    %0d %0d %0d  shamt:%0d  rd:%0d", state, PC, instruction, aluIn1, aluIn2, aluOut, shamt, rdId);
        if (isAUIPC    ) $display("STATE: %0d   PC:%3h %h  AUIPC                        ", state, PC, instruction);
        if (isOP_IMM_32) $display("STATE: %0d   PC:%3h %h  OP_IMM_32 %0d %0d %0d  rd:%0d", state, PC, instruction, aluIn1, aluIn2, aluOut, rdId);
        if (isSTORE    ) $display("STATE: %0d   PC:%3h %h  STORE                        ", state, PC, instruction);
        if (isSTORE_FP ) $display("STATE: %0d   PC:%3h %h  STORE_FP                     ", state, PC, instruction);
        if (isAMO      ) $display("STATE: %0d   PC:%3h %h  AMO                          ", state, PC, instruction);
        if (isOP       ) $display("STATE: %0d   PC:%3h %h  OP        %0d %0d %0d  rd:%0d", state, PC, instruction, aluIn1, aluIn2, aluOut, rdId);
        if (isLUI      ) $display("STATE: %0d   PC:%3h %h  LUI       rd:%0d    Uimm:%0h", state, PC, instruction, rdId, Uimm);
        if (isOP_32    ) $display("STATE: %0d   PC:%3h %h  OP_32     %0d %0d %0d  rd:%0d", state, PC, instruction, aluIn1, aluIn2, aluOut, rdId);
        if (isMADD     ) $display("STATE: %0d   PC:%3h %h  MADD                         ", state, PC, instruction);
        if (isMSUB     ) $display("STATE: %0d   PC:%3h %h  MSUB                         ", state, PC, instruction);
        if (isNMSUB    ) $display("STATE: %0d   PC:%3h %h  NMSUB                        ", state, PC, instruction);
        if (isNMADD    ) $display("STATE: %0d   PC:%3h %h  NMADD                        ", state, PC, instruction);
        if (isOP_FP    ) $display("STATE: %0d   PC:%3h %h  OP_FP                        ", state, PC, instruction);
        if (isBRANCH   ) $display("STATE: %0d   PC:%3h %h  BRANCH    %0d  %0d %0d %0d  Bimm:%0d ", state, PC, instruction, {{32{rs1[31]}}, rs1[31:0]}, {{32{rs2[31]}}, rs2[31:0]}, rs1, rs2, {{32{Bimm[31]}}, Bimm});
        if (isJALR     ) $display("STATE: %0d   PC:%3h %h  JALR                         ", state, PC, instruction);
        if (isJAL      ) $display("STATE: %0d   PC:%3h %h  JAL                          ", state, PC, instruction);
        if (isSYSTEM   ) $display("STATE: %0d   PC:%3h %h  SYSTEM                       ", state, PC, instruction);
   end

endmodule

module SOC (
    input  clk,       // system clock 
    input  reset      // reset button
    // input  RXD,        // UART receive
    // output TXD         // UART transmit
);

    wire [63:0]      mem_addr;
    wire [31:0]      mem_rdata;
    wire             mem_rstrb;

    wire  [63:0]     mem_daddr;  // data address
    wire [63:0]      mem_drdata; // data data
    wire             mem_drstrb; // data strobe

    Memory RAM(
        .clk(clk),
        .mem_addr(mem_addr), /* I MEM */
        .mem_rdata(mem_rdata),
        .mem_rstrb(mem_rstrb),
        .mem_daddr(mem_daddr), /* D MEM */
        .mem_drdata(mem_drdata),
        .mem_drstrb(mem_drstrb)
    );

    Processor CPU(
        .clk(clk),
        .reset(reset),		 
        .mem_addr(mem_addr), /* I MEM */
        .mem_rdata(mem_rdata),
        .mem_rstrb(mem_rstrb),
        .mem_daddr(mem_daddr), /* D MEM */
        .mem_drdata(mem_drdata),
        .mem_drstrb(mem_drstrb)
    );
endmodule
