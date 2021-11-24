import logging
from ifcopenshell.util.element import get_psets

from ada import Assembly, Part
from .reader_utils import get_parent, resolve_name
from ..concepts import IfcRef


def read_hierarchy(f, a: Assembly, ifc_ref: IfcRef):
    for product in f.by_type("IfcProduct"):
        parent, new_part = import_ifc_hierarchy(a, product, ifc_ref)
        if new_part is None:
            continue
        if parent is None:
            a.add_part(new_part)
        elif type(parent) is not Part:
            raise NotImplementedError()
        else:
            parent.add_part(new_part)


def import_ifc_hierarchy(assembly: Assembly, product, ifc_ref: IfcRef):
    pr_type = product.is_a()
    pp = get_parent(product)
    if pp is None:
        return None, None

    if pr_type not in ["IfcBuilding", "IfcSpace", "IfcBuildingStorey", "IfcSpatialZone"]:
        return None, None

    props = get_psets(product)
    name = product.Name
    if name is None:
        logging.debug(f'Name was not found for the IFC element "{product}". Will look for ref to name in props')
        name = resolve_name(props, product)

    new_part = Part(
        name,
        metadata=dict(original_name=name, props=props),
        guid=product.GlobalId,
        ifc_ref=ifc_ref,
        units=assembly.units,
    )

    pp_name = pp.Name
    if pp_name is None:
        pp_name = resolve_name(get_psets(pp), pp)
    if pp_name is None:
        return None, None
    parent = assembly.get_by_name(pp_name)
    return parent, new_part
