#!/usr/bin/python3

from elftools.elf.elffile import ELFFile
import glob
import binascii

def main():
    # for x in glob.glob("../../riscv-tests/isa/rv64ui-p-*"):
    # if (x.endswith('.dump')):
    #     continue
    x = "../../riscv-tests/isa/rv64ui-p-ld"
    with open(x, 'rb') as f:
        elffile = ELFFile(f)
        for section_idx in range(elffile.num_sections()):
            section = elffile.get_section(section_idx)
            if section.name == ".data":
                mem_data = section.data()
                with open("test_a/mem_data-%s" % x.split("/")[-1], "wb") as g:
                    g.write(b'\n'.join([binascii.hexlify(mem_data[i:i+8][::-1]) for i in range(0,len(mem_data),8)]))
            if section.name == ".text.init":
                mem_instr = section.data()
                with open("test_a/mem_instr-%s" % x.split("/")[-1], "wb") as g:
                    g.write(b'\n'.join([binascii.hexlify(mem_instr[i:i+4][::-1]) for i in range(0,len(mem_instr),4)]))

if __name__ == '__main__':
    main()
