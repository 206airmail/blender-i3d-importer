"""Adapter: convert a decoded i3d Shape (from i3d_shapes_models) into a
MeshData dataclass that the existing _build_mesh_datablock in importer.py
consumes. Self-contained: the MeshData definition lives here, so no
dependency on the legacy objx_parser module.

Mapping notes:
- Position/Normal/UV are direct field-by-field copies.
- Triangles in our reader are 1-based (per I3DTri.cs convention); MeshData
  stores 0-based indices. We subtract 1 here.
- MeshData stores per-corner (v_idx, vt_idx, vn_idx) tuples. In the binary
  format all three share the same index per vertex, so we replicate v_idx
  into all three slots (or None if the corresponding stream is missing).
- face_subsets maps each triangle index to the subset it belongs to. In the
  binary, each Subset describes [first_index, first_index+num_indices) over
  the global triangle array, so we just walk the Subsets list and fill the
  array.
- Vertex colors I3DVector4 -> RGBA tuple. Only emitted if the shape carries
  them.

What is intentionally NOT mapped here:
- BlendIndices / BlendWeights — those drive Merge-Group / Armature import
  paths that have their own _process_* passes in importer.py.
- Tangents — Blender derives those from the imported geometry.
- Material slot names (file_version >= 10) — not used by the existing
  _build_mesh_datablock path (it pulls materials from the XML side).
- Generic data — purpose-specific (vehicle-shader vertex inputs); handled
  by _process_merge_children.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class MeshData:
    """A parsed OBJx file as raw lists."""
    name: str = ""
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    uvs: List[Tuple[float, float]] = field(default_factory=list)
    normals: List[Tuple[float, float, float]] = field(default_factory=list)
    # faces: each face is a list of (vertex_idx, uv_idx, normal_idx) tuples,
    # 0-based; uv_idx/normal_idx may be None.
    faces: List[List[Tuple[int, int, int]]] = field(default_factory=list)
    # face_subsets: per face the 0-based subset index from the current usemtl
    # range. Faces before any usemtl line get subset 0 (default).
    face_subsets: List[int] = field(default_factory=list)
    # Number of subsets found (= max(face_subsets) + 1, or 0 if no faces).
    num_subsets: int = 0

    # Multi-UV: parallel to uvs (the same vt index in an f line applies to all
    # UV maps). In OBJx, vt, vt2, vt3, vt4 each have the same number of entries
    # when present.
    uvs2: List[Tuple[float, float]] = field(default_factory=list)
    uvs3: List[Tuple[float, float]] = field(default_factory=list)
    uvs4: List[Tuple[float, float]] = field(default_factory=list)

    # Vertex colors (RGBA): parallel to vertices, coming from the optional
    # 4 values in the v line. Either complete (len == len(vertices)) or empty.
    vertex_colors: List[Tuple[float, float, float, float]] = field(default_factory=list)



try:
    from .i3d_shapes_models import Shape
except ImportError:  # pragma: no cover  (standalone test convenience)
    from i3d_shapes_models import Shape


def shape_to_mesh_data(shape: Shape, name: str = None) -> MeshData:
    """Convert a decoded Shape into the MeshData dataclass used downstream.

    The `name` argument is mostly cosmetic (used as MeshData.name).
    """
    md = MeshData(name=name if name is not None else shape.name)

    # ---- Positions ----
    md.vertices = [(v.x, v.y, v.z) for v in shape.positions]

    # ---- Normals ----
    if shape.normals is not None:
        md.normals = [(n.x, n.y, n.z) for n in shape.normals]
    # else: leave [] -> downstream marks the face references as None for vn_idx

    # ---- UV channels ----
    if shape.uv_sets[0] is not None:
        md.uvs = [(uv.u, uv.v) for uv in shape.uv_sets[0]]
    if shape.uv_sets[1] is not None:
        md.uvs2 = [(uv.u, uv.v) for uv in shape.uv_sets[1]]
    if shape.uv_sets[2] is not None:
        md.uvs3 = [(uv.u, uv.v) for uv in shape.uv_sets[2]]
    if shape.uv_sets[3] is not None:
        md.uvs4 = [(uv.u, uv.v) for uv in shape.uv_sets[3]]

    # ---- Vertex colors ----
    if shape.vertex_colors is not None:
        md.vertex_colors = [(c.x, c.y, c.z, c.w) for c in shape.vertex_colors]

    # ---- Faces ----
    have_uv = bool(md.uvs)
    have_normals = bool(md.normals)

    # MeshData stores (v_idx, vt_idx, vn_idx) with vt_idx/vn_idx None if
    # absent. In the binary all three share the same per-vertex index.
    for tri in shape.triangles:
        i1 = tri.p1 - 1
        i2 = tri.p2 - 1
        i3 = tri.p3 - 1
        vt1 = i1 if have_uv else None
        vt2 = i2 if have_uv else None
        vt3 = i3 if have_uv else None
        vn1 = i1 if have_normals else None
        vn2 = i2 if have_normals else None
        vn3 = i3 if have_normals else None
        md.faces.append([(i1, vt1, vn1), (i2, vt2, vn2), (i3, vt3, vn3)])

    # ---- face_subsets ----
    # Subset.first_index / num_indices index into the global triangle array
    # (each triangle = 3 indices, so divide by 3 to get the triangle range).
    num_tris = len(shape.triangles)
    md.face_subsets = [0] * num_tris
    md.num_subsets = max(1, len(shape.subsets))
    for subset_idx, sub in enumerate(shape.subsets):
        first_tri = sub.first_index // 3
        last_tri = (sub.first_index + sub.num_indices) // 3  # exclusive
        # Defensive clamp — protects against corrupt subset descriptors.
        first_tri = max(0, min(first_tri, num_tris))
        last_tri = max(0, min(last_tri, num_tris))
        for t in range(first_tri, last_tri):
            md.face_subsets[t] = subset_idx

    return md
