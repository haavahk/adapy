from __future__ import annotations

import logging
import os
import pathlib
import pickle
import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ada.fem.results.common import FEAResult, FieldData, Mesh

_script_dir = pathlib.Path(__file__).parent.resolve().absolute()


ABA_IO = _script_dir / "aba_io.py"


def get_odb_data(odb_path, overwrite=False, use_aba_version=None):
    odb_path = pathlib.Path(odb_path)
    pickle_path = odb_path.with_suffix(".pckle")

    if pickle_path.exists() is False or overwrite is True:
        aba_ver = "abaqus" if use_aba_version is None else use_aba_version
        aba_exe_path = pathlib.Path(shutil.which(aba_ver))

        odb_path = pathlib.Path(odb_path)

        if os.path.isfile(pickle_path):
            os.remove(pickle_path)

        print(f'Extracting ODB data from "{odb_path.name}" using Abaqus/Python')

        backup_odb = odb_path.parent / f"{odb_path.stem}_backup.odb"
        if backup_odb.exists() is False:
            print(f'Copying a backup of the odb file to "{backup_odb}" in case python corrupts the odb file')
            shutil.copy(odb_path, backup_odb)

        res = subprocess.run([aba_exe_path, "python", ABA_IO, odb_path], cwd=ABA_IO.parent, capture_output=True)
        logging.info(str(res.stdout, encoding="utf-8"))
        stderr = str(res.stderr, encoding="utf-8")
        if stderr != "":
            logging.error(stderr)

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    return data


def read_odb_pckle_file(pickle_path: str | pathlib.Path) -> FEAResult:
    from ada.fem.formats.general import FEATypes
    from ada.fem.results.common import FEAResult

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    mesh = get_odb_instance_data(data["rootAssembly"]["instances"])
    fields = get_odb_frame_data(data["steps"])

    return FEAResult(name=pickle_path.stem, software=FEATypes.ABAQUS, mesh=mesh, results=fields)


def get_odb_field_data(field_name, field_data, frame_num):
    from ada.fem.results.field_data import ElementFieldData, NodalFieldData

    field_type, components, data = field_data

    if len(components) == 0:
        data_col = ["data"]
        data_format = [float]
    else:
        data_col = components
        data_format = len(components) * [float]

    if field_type == "ELEMENT_NODAL":
        dtype = np.dtype({"names": ElementFieldData.COLS + data_col, "formats": [int, int, int] + data_format})
        field_values = np.fromiter(yield_elem_nodal_data(data), dtype=dtype, count=len(data))
        return ElementFieldData(field_name, frame_num, components, values=field_values)
    elif field_type == "NODAL":
        dtype = np.dtype({"names": NodalFieldData.COLS + data_col, "formats": [int] + data_format})
        field_values = np.fromiter(yield_nodal_data(data), dtype=dtype, count=len(data))
        return NodalFieldData(field_name, frame_num, components, field_values)
    else:
        raise NotImplementedError()


def get_odb_frame_data(steps: dict) -> list[FieldData]:
    frame_num = 0
    fields = []
    for step_name, step_data in dict(sorted(steps.items(), key=lambda x: x[1]["totalTime"])).items():
        for frame in step_data["frames"]:
            for key, value in frame.items():
                field = get_odb_field_data(key, value["values"], frame_num)
                fields.append(field)
            frame_num += 1

    return fields


def get_odb_instance_data(instances) -> Mesh:
    from ada.fem.formats.abaqus.elem_shapes import abaqus_el_type_to_ada
    from ada.fem.formats.general import FEATypes
    from ada.fem.results.common import ElementBlock, ElementType, Mesh, Nodes

    if len(instances) > 1:
        raise NotImplementedError("Multi-instances results are not yet supported")

    instance = instances[0]

    ids, coords = zip(*instance["nodes"])
    el_ids, el_type_array, nodes_connectivity, sec_cat = zip(*instance["elements"])
    el_type_set = set(el_type_array)
    if len(el_type_set) != 1:
        raise NotImplementedError("Mixed element sets not yet supported")

    el_type = el_type_array[0]
    shape = abaqus_el_type_to_ada(el_type)
    elem_type = ElementType(type=shape, source_software=FEATypes.ABAQUS, source_type=el_type)
    el_block = ElementBlock(
        type=elem_type, nodes=np.array(nodes_connectivity, dtype=int), identifiers=np.array(el_ids, dtype=int)
    )
    el_blocks = [el_block]
    nodes = Nodes(coords=np.array(coords, dtype=float), identifiers=np.array(ids, dtype=int))
    return Mesh(elements=el_blocks, nodes=nodes)


def yield_elem_nodal_data(data):
    for x in data:
        spn = x["sec_p_num"]
        if isinstance(spn, dict):
            spn = -1
        if isinstance(x["data"], list) is False:
            x["data"] = [x["data"]]

        yield x["elementLabel"], spn, x["nodeLabel"], *x["data"]


def yield_nodal_data(data):
    for x in data:
        yield x[0], *x[1]
