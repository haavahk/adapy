from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, Union

import numpy as np

from ada.core.utils import thread_this
from ada.occ.exceptions.geom_creation import (
    UnableToBuildNSidedWires,
    UnableToCreateSolidOCCGeom,
    UnableToCreateTesselationFromSolidOCCGeom,
)
from ada.visualize.concept import PolyModel
from ada.visualize.renderer_occ import occ_shape_to_faces

from .config import ExportConfig

if TYPE_CHECKING:
    from ada import Beam, PipeSegElbow, PipeSegStraight, Plate, Shape, Wall


def list_of_obj_to_json(
    list_of_all_objects: Iterable[Union[Beam, Plate, Wall, PipeSegElbow, PipeSegStraight, Shape]],
    obj_num: int,
    all_obj_num: int,
    export_config: ExportConfig,
):
    from ada import Pipe

    id_map = dict()
    for obj in list_of_all_objects:
        obj_num += 1
        if isinstance(obj, Pipe):
            for seg in obj.segments:
                res = obj_to_json(seg, export_config)
                if res is None:
                    continue
                id_map[seg.guid] = res.to_dict()
                print(f'Exporting "{obj.name}" ({obj_num} of {all_obj_num})')
        else:
            res = obj_to_json(obj, export_config)
            if res is None:
                continue
            id_map[obj.guid] = res.to_dict()
            print(f'Exporting "{obj.name}" ({obj_num} of {all_obj_num})')

    return id_map


def ifc_elem_to_json(obj: Shape, export_config: ExportConfig = ExportConfig()):
    import ifcopenshell.geom

    a = obj.get_assembly()
    ifc_f = a.get_ifc_source_by_name(obj.ifc_ref.source_ifc_file)
    ifc_elem = ifc_f.by_guid(obj.guid)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_PYTHON_OPENCASCADE, False)
    settings.set(settings.SEW_SHELLS, False)
    settings.set(settings.WELD_VERTICES, False)
    settings.set(settings.INCLUDE_CURVES, False)
    settings.set(settings.USE_WORLD_COORDS, False)
    settings.set(settings.VALIDATE_QUANTITIES, False)

    geom = obj.ifc_ref.get_ifc_geom(ifc_elem, settings)
    obj_position = np.array(geom.geometry.verts, dtype=float)
    poly_indices = np.array(geom.geometry.faces, dtype=int)
    normals = geom.geometry.normals if len(geom.geometry.normals) != 0 else None
    mat0 = geom.geometry.materials[0]
    colour = [*mat0.diffuse, mat0.transparency]
    return obj_position, poly_indices, normals, colour


def occ_geom_to_poly_mesh(
    obj: Union[Beam, Plate, Wall, PipeSegElbow, PipeSegStraight, Shape], export_config: ExportConfig = ExportConfig()
):
    geom = obj.solid
    obj_position, poly_indices, normals, _ = occ_shape_to_faces(
        geom,
        export_config.quality,
        export_config.render_edges,
        export_config.parallel,
    )
    return obj_position, poly_indices, normals, [*obj.colour_norm, obj.opacity]


def obj_to_json(
    obj: Union[Beam, Plate, Wall, PipeSegElbow, PipeSegStraight, Shape], export_config: ExportConfig = ExportConfig()
) -> Union[PolyModel, None]:
    if obj.ifc_ref is not None and export_config.ifc_skip_occ is True:
        position, indices, normals, colour = ifc_elem_to_json(obj)
    else:
        try:
            obj_position, poly_indices, normals, colour = occ_geom_to_poly_mesh(obj, export_config)
        except (UnableToBuildNSidedWires, UnableToCreateTesselationFromSolidOCCGeom, UnableToCreateSolidOCCGeom) as e:
            logging.error(e)
            return None

        obj_buffer_arrays = np.concatenate([obj_position, normals], 1)
        buffer, indices = np.unique(obj_buffer_arrays, axis=0, return_index=False, return_inverse=True)
        x, y, z, nx, ny, nz = buffer.T
        position = np.array([x, y, z]).T
        normals = np.array([nx, ny, nz]).T

    return PolyModel(obj.guid, indices, position, normals, colour)


def id_map_using_threading(list_in, threads: int):
    # obj = list_in[0]
    # obj_str = json.dumps(obj)
    # serialize_evaluator(obj)
    res = thread_this(list_in, obj_to_json, threads)
    print(res)
    return res
