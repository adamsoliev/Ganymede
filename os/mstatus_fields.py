import sys

def print_mstatus_fields(mstatus_value):
    mstatus_fields = {
        'SD': 63,
        'MBE': 37,
        'SBE': 36,
        'SXL': (35, 34),
        'UXL': (33, 32),
        'TSR': 22,
        'TW': 21,
        'TVM': 20,
        'MXR': 19,
        'SUM': 18,
        'MPRV': 17,
        'XS': (16, 15),
        'FS': (14, 13),
        'MPP': (12, 11),
        'VS': (10, 9),
        'SPP': 8,
        'MPIE': 7,
        'UBE': 6,
        'SPIE': 5,
        'MIE': 3,
        'SIE': 1
    }

    max_field_len = max(len(field) for field in mstatus_fields.keys())

    for field, indexes in mstatus_fields.items():
        
        if isinstance(indexes, int):
            value = (mstatus_value >> indexes) & 1
            print(f"{field.rjust(max_field_len)}: {bin(value)[2:]}")
        else:
            start, end = indexes
            # value = (mstatus_value >> start) & ((1 << (end - start + 1)) - 1)
            value = (mstatus_value >> end) & 3
            binary_value = bin(value)[2:].zfill(start - end + 1)
            # print(f"{field}: {binary_value}")
            print(f"{field.rjust(max_field_len)}: {binary_value}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <mstatus_value>")
        sys.exit(1)

    input_value = sys.argv[1]

    try:
        if input_value.startswith('0b'):
            mstatus_value = int(input_value, 2) 
        elif input_value.startswith('0x'):
            mstatus_value = int(input_value, 16)
        else:
            mstatus_value = int(input_value)
        print_mstatus_fields(mstatus_value)
    except ValueError:
        print("Invalid/missing input")
        sys.exit(1)

