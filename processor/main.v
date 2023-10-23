/*
register file   - read two registers at each cycle, and optionally write-back one
ALU             - do computation
decoder         - generate all required internal signals from the bit pattern of the current instruction
*/

// module SOC (
//     input  CLK,        // system clock 
//     input  RESET,      // reset button
//     output [4:0] LEDS, // system LEDs
//     input  RXD,        // UART receive
//     output TXD         // UART transmit
// );

//     // A blinker that counts on 5 bits, wired to the 5 LEDs
//     reg [4:0] count = 0;
//     always @(posedge CLK) begin
//         count <= count + 1;
//     end

//     assign LEDS = count;
//     assign TXD  = 1'b0; // not used for now   
// endmodule


/**
 * Step 3: Display a led pattern "animation" stored in BRAM.
 * DONE*
 */

module SOC (
    input  CLK,        // system clock 
    input  RESET,      // reset button
    output [4:0] LEDS, // system LEDs
    input  RXD,        // UART receive
    output TXD         // UART transmit
);

   reg [4:0] PC = 0;
   reg [4:0] MEM [0:20];
   initial begin
       MEM[0]  = 5'b00000;
       MEM[1]  = 5'b00001;
       MEM[2]  = 5'b00010;
       MEM[3]  = 5'b00100;
       MEM[4]  = 5'b01000;
       MEM[5]  = 5'b10000;
       MEM[6]  = 5'b10001;
       MEM[7]  = 5'b10010;
       MEM[8]  = 5'b10100;
       MEM[9]  = 5'b11000;
       MEM[10] = 5'b11001;
       MEM[11] = 5'b11010;
       MEM[12] = 5'b11100;
       MEM[13] = 5'b11101;
       MEM[14] = 5'b11110;
       MEM[15] = 5'b11111;
       MEM[16] = 5'b11110;
       MEM[17] = 5'b11100;
       MEM[18] = 5'b11000;
       MEM[19] = 5'b10000;
       MEM[20] = 5'b00000;       
   end

   reg [4:0] leds = 0;
   assign LEDS=leds;

   always @(posedge CLK) begin
      leds <= MEM[PC];
      PC <= (PC==20) ? 0 : (PC+1);
   end

   assign TXD  = 1'b0; // not used for now   
endmodule