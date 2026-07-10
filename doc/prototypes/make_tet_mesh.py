"""Generate a structured tetrahedral test mesh in NASTRAN (.nas) format.

Splits each cube of an Nx x Ny x Nz grid into 6 tets (Kuhn triangulation,
conforming across cube faces). Card format matches the free-field
comma-separated GRID/CTETRA cards expected by gridTet.read_nas_file_pd.

Usage: python make_tet_mesh.py [Nx Ny Nz out.nas]
"""

import sys

import numpy as np
from itertools import permutations

Nx, Ny, Nz = 12, 12, 8
out = "tet_test_mesh.nas"
if len(sys.argv) == 5:
    Nx, Ny, Nz, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]


def node_id(i, j, k):
    return i * (Ny + 1) * (Nz + 1) + j * (Nz + 1) + k + 1  # 1-based


with open(out, "w") as fh:
    fh.write("$ structured Kuhn tet mesh\n")
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            for k in range(Nz + 1):
                fh.write(f"GRID,{node_id(i, j, k)},0,{float(i)},{float(j)},{float(k)}\n")

    def parity(p):
        inv = sum(1 for a in range(3) for b in range(a + 1, 3) if p[a] > p[b])
        return inv % 2

    eid = 1
    unit = [np.eye(3, dtype=int)[d] for d in range(3)]
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                base = np.array([i, j, k])
                for perm in permutations(range(3)):
                    verts = [base.copy()]
                    v = base.copy()
                    for d in perm:
                        v = v + unit[d]
                        verts.append(v.copy())
                    if parity(perm):  # flip odd permutations -> detJ > 0
                        verts[2], verts[3] = verts[3], verts[2]
                    ids = [node_id(*vv) for vv in verts]
                    fh.write(f"CTETRA,{eid},1,{ids[0]},{ids[1]},{ids[2]},{ids[3]}\n")
                    eid += 1
    fh.write("ENDDATA\n")

print(f"Wrote {out}: {(Nx+1)*(Ny+1)*(Nz+1)} nodes, {6*Nx*Ny*Nz} tets")
