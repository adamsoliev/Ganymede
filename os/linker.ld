OUTPUT_ARCH( "riscv" )

ENTRY( _entry )

MEMORY {
  ram   (wxa!ri) : ORIGIN = 0x80000000, LENGTH = 1M
}

SECTIONS {

  . = 0x80000000;

  .text : {
    *(.text.init) *(.text .text.*)
  } 
   PROVIDE(_global_pointer = .);
  .rodata : {
    *(.rodata .rodata.*)
  }

  .data : {
    . = ALIGN(4096);
    *(.sdata .sdata.*) *(.data .data.*)
  }

  .bss :{
    *(.sbss .sbss.*) *(.bss .bss.*)
  }

  PROVIDE(_stack = . + 0x80000);
}

/*

            +---------------------------+
            |        Unallocated        |
            +---------------------------+
            |      (Heap and Stack)     |
            +---------------------------+
            |        .bss               |
            +---------------------------+
            |        .data              |
            +---------------------------+
            |        .rodata            |
            +---------------------------+
            |        .text              |
0x80000000  +---------------------------+
            |                           |
            |         I/O devices       |
            |                           |
0x00000000  +---------------------------+
*/