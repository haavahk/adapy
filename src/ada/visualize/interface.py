from __future__ import annotations

from typing import TYPE_CHECKING

import ifcopenshell.geom
import numpy as np

from ada.visualize.concept import ObjectMesh, PartMesh, VisMesh

if TYPE_CHECKING:
    from ada import Part


def part_to_vis_mesh2(part: Part, auto_sync_ifc_store=True, cpus: int = None) -> VisMesh:
    ifc_store = part.get_assembly().ifc_store
    if auto_sync_ifc_store:
        ifc_store.sync()

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_PYTHON_OPENCASCADE, False)
    settings.set(settings.SEW_SHELLS, False)
    settings.set(settings.WELD_VERTICES, True)
    settings.set(settings.INCLUDE_CURVES, False)
    settings.set(settings.USE_WORLD_COORDS, True)
    settings.set(settings.VALIDATE_QUANTITIES, False)

    iterator = ifc_store.get_ifc_geom_iterator(settings, cpus=cpus)
    iterator.initialize()
    id_map = dict()

    while True:
        shape = iterator.get()
        if shape:
            obj_mesh = product_to_obj_mesh(shape)
            id_map[shape.guid] = obj_mesh

        if not iterator.next():
            break

    pm = PartMesh(name=part.name, id_map=id_map)
    meta = {
        p.guid: (p.name, p.parent.name if p.parent is not None else "*")
        for p in part.get_all_subparts(include_self=True)
    }
    parts_d = {p.guid: (p.name, p.parent.guid) for p in part.get_all_physical_objects()}
    meta.update(parts_d)
    return VisMesh(part.name, world=[pm], meta=meta)


def product_to_obj_mesh(shape: ifcopenshell.ifcopenshell_wrapper.TriangulationElement) -> ObjectMesh:
    geometry = shape.geometry
    vertices = np.array(geometry.verts, dtype="float32").reshape(int(len(geometry.verts) / 3), 3)
    faces = np.array(geometry.faces, dtype=int)
    normals = np.array(geometry.normals) if len(geometry.normals) != 0 else None

    if normals is not None and len(normals) > 0:
        normals = normals.astype(dtype="float32").reshape(int(len(normals) / 3), 3)

    mats = geometry.materials
    if len(mats) == 0:
        colour = [1.0, 0.0, 0.0, 1.0]
    else:
        mat0 = mats[0]
        opacity = 1.0 - mat0.transparency
        colour = [*mat0.diffuse, opacity]

    return ObjectMesh(shape.guid, faces, vertices, normals, color=colour)
