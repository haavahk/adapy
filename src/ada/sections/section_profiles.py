from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

from ada.sections.categories import BaseTypes


@dataclass
class Profile(ABC):
    """Section profile data"""

    @property
    def sec_str_formatting(self) -> str:
        return "x".join([f"{v}"for v in self.sec_str_vars])

    @property
    @abstractmethod
    def sec_str_vars(self) -> Iterable:
        """Section string variables"""

    @property
    @abstractmethod
    def base_type(self) -> str:
        """Returns the profile base type"""

    @property
    def sec_str(self) -> str:
        """Returns profile section string"""
        return self.base_type + self.sec_str_formatting


@dataclass
class CircularProfile(Profile):
    r: float

    @property
    def base_type(self) -> str:
        return BaseTypes.CIRCULAR

    @property
    def sec_str_vars(self) -> Iterable:
        return (self.r,)


@dataclass
class TubularProfile(CircularProfile):
    t: float

    @property
    def base_type(self) -> str:
        return BaseTypes.TUBULAR

    @property
    def sec_str_vars(self) -> Iterable:
        return self.r, self.t


@dataclass
class FlatbarProfile(Profile):
    """Flatbar profile"""
    h: float
    w_top: float
    w_btn: float

    @property
    def base_type(self) -> str:
        return BaseTypes.FLATBAR

    @property
    def sec_str_vars(self) -> Iterable:
        return self.h, self.w_top


@dataclass
class BoxProfile(FlatbarProfile):
    t_w: float
    t_ftop: float
    t_fbtn: float

    @property
    def base_type(self) -> str:
        return BaseTypes.BOX

    @property
    def sec_str_vars(self) -> Iterable:
        return *super(BoxProfile, self).sec_str_vars, self.t_w, self.t_ftop


@dataclass
class ChannelProfile(Profile):
    h: float
    w_top: float
    w_btn: float
    t_w: float
    t_fbtn: float

    @property
    def base_type(self) -> str:
        return BaseTypes.CHANNEL

    @property
    def sec_str_vars(self) -> Iterable:
        return (self.h,)


@dataclass
class IProfile(BoxProfile):
    """I-Profile"""

    @property
    def base_type(self) -> str:
        return BaseTypes.IPROFILE


@dataclass
class TProfile(BoxProfile):
    """T-Profile"""

    @property
    def base_type(self) -> str:
        return BaseTypes.TPROFILE


@dataclass
class LProfile(ChannelProfile):
    """L-Profile"""

    @property
    def base_type(self) -> str:
        return BaseTypes.LPROFILE
