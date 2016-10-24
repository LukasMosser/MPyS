import numpy as np


def import_sgems_dat_file(file_path):
    """

    Imports an SGEMS .dat file. See SGeMS documentation for file, format specification.

    """
    with open(file_path, "r") as f:
        img = []
        for i, row in enumerate(f):
            if i == 0:
                row_vals = row.split(" ")
                nx = int(row_vals[0])
                ny = int(row_vals[1])
                nz = int(row_vals[2])
            elif i >2:
                stripped_vals = [val.strip("\n") for val in row.split(" ")]
                legit_values = [val for val in stripped_vals if val not in ["", " ", "\""]]
                node_value = int(legit_values[0])
                img.append(node_value)
        img = np.array(img).reshape(ny, nx).astype(np.int32)
    return img