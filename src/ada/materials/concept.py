from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, KW_ONLY
from typing import TYPE_CHECKING

from ada.base.non_physical_objects import Backend

from .metals import CarbonSteel

if TYPE_CHECKING:
    from ada.ifc.concepts import IfcRef
    from ada.materials.metals.base_models import MaterialModel


@dataclass
class Material(Backend):
    """The base material class. Currently only supports Metals"""
    id: int
    model: MaterialModel

    def equal_props(self):
        """Has material equal props"""

    def __hash__(self):
        return hash(self.guid)
