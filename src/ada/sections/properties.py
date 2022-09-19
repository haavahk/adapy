from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

import numpy as np

from .categories import SectionCat
from .concept import GeneralProperties
from .section_profiles import CircularProfile, TubularProfile, FlatbarProfile, BoxProfile, ChannelProfile, \
    IProfile, TProfile, LProfile

if TYPE_CHECKING:
    from .section_profiles import Profile


# List of documents the various formulas are based upon
#
#   * StructX.com (https://www.structx.com/geometric_properties.html)
#   * DNVGL. (2011). Appendix B Section properties & consistent units Table of Contentsec. I.
#   * W. Beitz, K.H. Küttner: "Dubbel, Taschenbuch für den Maschinenbau" 17. Auflage (17th ed.)
#     Springer-Verlag 1990
#   * Arne Selberg: "Stålkonstruksjoner" Tapir 1972
#   * sec. Timoshenko: "Strength of Materials, Part I, Elementary Theory and Problems" Third Edition 1995 D.
#     Van Nostrand Company Inc.


def calculate_general_properties(profile: Profile) -> Union[None, GeneralProperties]:
    """Calculations of cross section properties are based on different sources of information."""
    # bt = SectionCat.BASETYPES
    section_map = {
        CircularProfile: calc_circular,
        IProfile: calc_isec,
        BoxProfile: calc_box,
        TubularProfile: calc_tubular,
        ChannelProfile: calc_channel,
        FlatbarProfile: calc_flatbar,
        TProfile: calc_isec,
        LProfile: calc_angular
    }
    calculator = section_map.get(profile.__class__)
    # base_type = SectionCat.get_shape_type(section)

    # if base_type == bt.GENERAL:
    #     logging.error("Re-Calculating a general section")
    #     return None

    # calc_func = section_map.get(base_type, None)

    if calculator is None:
        raise Exception(
            f'Section profile type "{profile.base_type}" is not yet supported in the cross section parameter calculations'
        )

    return calculator(profile)


def calc_box(sec: BoxProfile) -> GeneralProperties:
    """Calculate box cross section properties"""

    sfy = 1.0
    sfz = 1.0

    area = sec.w_btn * sec.t_fbtn + sec.w_top * sec.t_ftop + sec.t_w * (sec.h - (sec.t_fbtn + sec.t_ftop)) * 2

    by = sec.w_top
    tt = sec.t_ftop
    tb = sec.t_fbtn
    ty = sec.t_w
    hz = sec.h

    a = tb / 2
    b = (hz + tb - tt) / 2
    c = hz - tt / 2
    d = sec.h - sec.t_fbtn - sec.t_ftop
    e = by * tb
    f = by * tt
    g = ty * d

    partial_area = e + f + 2 * g
    h = (e * a + f * c + 2 * b * g) / partial_area
    ha = sec.h - (sec.t_fbtn + sec.t_ftop) / 2.0
    hb = sec.w_top - sec.t_w

    ix = 4 * (ha * hb) ** 2 / (hb / tb + hb / ty + 2 * ha / ty)
    iy = (by * (tb**3 + tt**3) + 2 * ty * d**3) / 12 + e * (h - a) ** 2 + f * (c - h) ** 2 + 2 * g * (b - h) ** 2

    iz = ((sec.t_fbtn + sec.t_ftop) * sec.w_top**3 + 2 * d * sec.t_w**3) / 12 + (g * hb**2) / 2
    iyz = 0
    wxmin = ix * (hb + ha) / (ha * hb)
    wymin = iy / max(sec.h - h, h)
    wzmin = 2 * iz / sec.w_top
    sy = e * (h - a) + ty * (h - tb) ** 2
    Sz = (sec.t_fbtn + sec.t_ftop) * sec.w_top**2 / 8 + g * hb / 2
    shary = (iz / Sz) * 2 * sec.t_w * sfy
    sharz = (iy / sy) * 2 * ty * sfz
    shceny = 0
    shcenz = c - h - sec.t_fbtn * ha / (sec.t_fbtn + sec.t_ftop)
    cy = sec.w_top / 2
    cz = h
    return GeneralProperties(
        area=area,
        ix=ix,
        iy=iy,
        iz=iz,
        iyz=iyz,
        wxmin=wxmin,
        wymin=wymin,
        wzmin=wzmin,
        shary=shary,
        sharz=sharz,
        shceny=shceny,
        shcenz=shcenz,
        sy=sy,
        sz=Sz,
        sfy=sfy,
        sfz=sfz,
        cy=cy,
        cz=cz,
        # parent=sec,
    )


def calc_isec(sec: IProfile) -> GeneralProperties:
    """Calculate I/H cross section properties"""

    sfy = 1.0
    sfz = 1.0
    hz = sec.h
    bt = sec.w_top
    tt = sec.t_ftop
    ty = sec.t_w
    bb = sec.w_btn
    tb = sec.t_fbtn

    area = bt * tt + ty * (hz - (tb + tt)) + bb * tb
    hw = hz - tt - tb
    a = tb + hw + tt / 2
    b = tb + hw / 2
    c = tb / 2

    z = (bt * tt * a + hw * ty * b + bb * tb * c) / area

    tra = (bt * tb**3) / 12 + bt * tt * (hz - tt / 2 - z) ** 2
    trb = (ty * hw**3) / 12 + ty * hw * (tb + hw / 2 - z) ** 2
    trc = (bb * tb**3) / 12 + bb * tb * (tb / 2 - z) ** 2

    if tt == ty and tt == tb:
        ix = (tt**3) * (hw + bt + bb - 1.2 * tt) / 3
        wxmin = ix / tt
    else:
        ix = 1.3 * (bt * tt**3 + hw * ty**3 + bb * tb**3) / 3
        wxmin = ix / max(tt, ty, tb)

    iy = tra + trb + trc
    iz = (tb * bb**3 + hw * ty**3 + tt * bt**3) / 12
    iyz = 0
    wymin = iy / max(hz - z, z)
    wzmin = 2 * iz / max(bb, bt)

    # sy should be checked. Confer older method implementation.
    # sy = sum(x_i * A_i)
    # sy = (((tt * bt) ** 2) * (hw / 2 + tt / 2)) * 2
    sy = iy / (sec.w_top / 2)

    # sy = (sec.t_w*sec.h/2)(sec.h/2)
    sz = (tt * bt**2 + tb * bb**2 + hw * ty**2) / 8
    shary = (iz / sz) * (tb + tt) * sfy
    sharz = (iy / sy) * ty * sfz
    shceny = 0
    shcenz = ((hz - tt / 2) * tt * bt**3 + (tb**2) * (bb**3) / 2) / (tt * bt**3 + tb * bb**3) - z
    cy = bb / 2
    cz = z

    return GeneralProperties(
        area=area,
        ix=ix,
        iy=iy,
        iz=iz,
        iyz=iyz,
        wxmin=wxmin,
        wymin=wymin,
        wzmin=wzmin,
        shary=shary,
        sharz=sharz,
        shceny=shceny,
        shcenz=shcenz,
        sy=sy,
        sz=sz,
        sfy=1,
        sfz=1,
        cy=cy,
        cz=cz,
        # parent=sec,
    )


def calc_angular(sec: LProfile) -> GeneralProperties:
    """Calculate L cross section properties"""

    # rectangle A properties (web)
    a_w = sec.t_w
    a_h = sec.h - sec.t_fbtn
    a_dy = a_w / 2
    a_dz = sec.t_fbtn + a_h / 2
    a_area = a_w * a_h

    # rectangle B properties (flange)
    b_w = sec.w_btn
    b_h = sec.t_fbtn
    b_dy = b_w / 2
    b_dz = b_h / 2
    b_area = b_h * b_w

    # Find centroid
    c_y = (a_area * a_dy + b_area * b_dy) / (a_area + b_area)
    c_z = (a_area * a_dz + b_area * b_dz) / (a_area + b_area)

    # c_z_opp = sec.h - c_z

    a_dcy = a_dy - c_y
    b_dcy = b_dy - c_y

    a_dcz = a_dz - c_z
    b_dcz = b_dz - c_z

    # Iz_a + A_a*dcy_a**2

    Iz_a = (1 / 12) * a_h * a_w**3 + a_area * a_dcy**2
    Iz_b = (1 / 12) * b_h * b_w**3 + b_area * b_dcy**2
    iz = Iz_a + Iz_b

    Iy_a = (1 / 12) * a_w * a_h**3 + a_area * a_dcz**2
    Iy_b = (1 / 12) * b_w * b_h**3 + b_area * b_dcz**2
    iy = Iy_a + Iy_b

    posweb = False

    r = 0

    hz = sec.h
    ty = sec.t_w
    tz = sec.t_fbtn
    by = sec.w_btn

    sfy = 1.0
    sfz = 1.0
    hw = hz - tz
    b = tz - hw / 2.0
    c = tz / 2.0
    piqrt = np.arctan(1.0)
    area = ty * hw + by * tz + (1 - piqrt) * r**2
    y = (hw * ty**2 + tz * by**2) / (2 * area)
    z = (hw * b * ty + tz * by * c) / area
    d = 6 * r + 2 * (ty + tz - np.sqrt(4 * r * (2 * r + ty + tz) + 2 * ty * tz))
    e = hw + tz - z
    f = hw - e
    ri = y - ty
    rj = by - y
    rk = ri + 0.5 * ty
    rl = z - c

    if tz >= ty:
        h = hw
    else:
        raise ValueError("Currently not implemented this yet")

    ix = (1 / 3) * (by * tz**3 + (hz - tz) * ty**3)
    iyz = (rl * tz / 2) * (y**2 - rj**2) - (rk * ty / 2) * (e**2 - f**2)

    wxmin = ix / d
    wymin = iy / max(z, hz - h)
    wzmin = iz / max(y, rj)
    sy = (ty * e**2) / 2
    sz = (tz * rj**2) / 2
    shary = (iz * tz / sz) * sfy
    sharz = (iy * tz / sy) * sfz

    if posweb:
        iyz = -iyz
        shceny = rk
        cy = by - y
    else:
        shceny = -rk
        cy = y
    shcenz = -rl
    cz = z

    return GeneralProperties(
        area=area,
        ix=ix,
        iy=iy,
        iz=iz,
        iyz=iyz,
        wxmin=wxmin,
        wymin=wymin,
        wzmin=wzmin,
        shary=shary,
        sharz=sharz,
        shceny=shceny,
        shcenz=shcenz,
        sy=sy,
        sz=sz,
        sfy=1,
        sfz=1,
        cy=cy,
        cz=cz,
        # parent=sec,
    )


def calc_tubular(sec: TubularProfile) -> GeneralProperties:
    """Calculate Tubular cross section properties"""

    t = sec.t
    sfy = 1.0
    sfz = 1.0

    dy = sec.r * 2
    di = dy - 2 * t
    area = np.pi * sec.r**2 - np.pi * (sec.r - t) ** 2
    ix = 0.5 * np.pi * ((dy / 2) ** 4 - (di / 2) ** 4)
    iy = ix / 2
    iz = iy
    iyz = 0
    wxmin = 2 * ix / dy
    wymin = 2 * iy / dy
    wzmin = 2 * iz / dy
    sy = (dy**3 - di**3) / 12
    sz = sy
    shary = (2 * iz * t / sy) * sfy
    sharz = (2 * iy * t / sz) * sfz
    shceny = 0
    shcenz = 0
    cy = 0.0
    cz = 0.0

    return GeneralProperties(
        area=area,
        ix=ix,
        iy=iy,
        iz=iz,
        iyz=iyz,
        wxmin=wxmin,
        wymin=wymin,
        wzmin=wzmin,
        shary=shary,
        sharz=sharz,
        shceny=shceny,
        shcenz=shcenz,
        sy=sy,
        sz=sz,
        sfy=1,
        sfz=1,
        cy=cy,
        cz=cz,
        # parent=sec,
    )


def calc_flatbar(sec: FlatbarProfile) -> GeneralProperties:
    """Flatbar (not supporting unsymmetric profile)"""
    w = sec.w_btn
    hz = sec.h

    a = 0.0
    h = hz * w / (2 * w)
    b = w / 2
    d = 0

    sfy = 1.0
    sfz = 1.0

    area = w * hz
    iy = w * hz**3 / 12
    iz = hz * w**3 / 12

    bm = 2 * w * hz**2 / (hz**2 + area**2)
    wymin = iy / max(h, d)
    wzmin = 2 * iz / max(w, w)
    iyz = 0.0
    if hz == bm:
        ca = 0.141
        cb = 0.208
        ix = ca * hz**4
        wxmin = cb * hz**3
    elif hz < bm:
        cn = bm / hz
        ca = (1 - 0.63 / cn + 0.052 / cn**5) * 3
        cb = ca / (1 - 0.63 / (1 + cn**3))
        ix = ca * bm * hz**3
        wxmin = cb * bm * hz**2
    else:
        cn = hz / bm
        ca = (1 - 0.63 / cn + 0.052 / cn**5) * 3
        cb = ca / (1 - 0.63 / (1 + cn**3))
        ix = ca * hz * bm**3
        wxmin = cb * hz * bm**3

    sy = (w * h**2) / 2 + (b - w / 2) * (h**2) / 3
    sz = hz * ((w**2) / 8 + a * (w / 4 + a / 6))

    shary = iz * hz * sfy / sz
    sharz = 2 * iy * b * sfz / sy

    shceny = 0.0
    shcenz = 0.0
    cy = w / 2
    cz = hz

    return GeneralProperties(
        area=area,
        ix=ix,
        iy=iy,
        iz=iz,
        iyz=iyz,
        wxmin=wxmin,
        wymin=wymin,
        wzmin=wzmin,
        shary=shary,
        sharz=sharz,
        shceny=shceny,
        shcenz=shcenz,
        sy=sy,
        sz=sz,
        sfy=sfy,
        sfz=sfz,
        cy=cy,
        cz=cz,
        # parent=sec,
    )


def calc_channel(sec: ChannelProfile) -> GeneralProperties:
    """Calculate section properties of a channel profile"""
    posweb = False
    hz = sec.h
    ty = sec.t_w
    tz = sec.t_fbtn
    by = sec.w_btn
    sfy = 1.0
    sfz = 1.0

    a = hz - 2 * tz
    area = 2 * by * tz + a * ty
    y = (2 * tz * by**2 + a * ty**2) / (2 * area)
    iy = (ty * a**3) / 12 + 2 * ((by * tz**3) / 12 + by * tz * ((a + tz) / 2) ** 2)

    if tz == ty:
        ix = ty**3 * (2 * by + a - 2.6 * ty) / 3
        wxmin = ix / iy
    else:
        ix = 1.12 * (2 * by * tz**3 + a * ty**3) / 3
        wxmin = ix / max(tz, ty)

    iz = 2 * ((tz * by**3) / 12 + tz * by * (by / 2 - y) ** 2) + (a * ty**3) / 12 + a * ty * (y - ty / 2) ** 2
    iyz = 0
    wymin = 2 * iy / hz
    wzmin = iz / max(by - y, y)
    sy = by * tz * (tz + a) / 2 + (ty * a**2) / 8
    sz = tz * (by - y) ** 2

    shary = (iz / sz) * (2 * tz) * sfy
    sharz = (iy / sy) * ty * sfz

    if tz == ty:
        q = ((by - ty / 2) ** 2) * ((hz - tz) ** 2) * tz / 4 * iy
    else:
        q = ((by - ty / 2) ** 2) * tz / (2 * (by - ty / 2) * tz + (hz - tz) * ty / 3)

    if posweb:
        shceny = y - ty / 2 + q
        cy = by
    else:
        shceny = -(y - ty / 2 + q)
        cy = by - y

    cz = hz / 2
    shcenz = 0

    return GeneralProperties(
        area=area,
        ix=ix,
        iy=iy,
        iz=iz,
        iyz=iyz,
        wxmin=wxmin,
        wymin=wymin,
        wzmin=wzmin,
        shary=shary,
        sharz=sharz,
        shceny=shceny,
        shcenz=shcenz,
        sy=sy,
        sz=sz,
        sfy=sfy,
        sfz=sfz,
        cy=cy,
        cz=cz,
        # parent=sec,
    )


def calc_circular(sec: CircularProfile) -> GeneralProperties:
    sfy = 1.0
    sfz = 1.0
    iyz = 0.0

    area = np.pi * sec.r**2
    iy = (np.pi * sec.r**4) / 4
    iz = iy
    ix = 0.5 * np.pi * sec.r**4
    wymin = 0.25 * np.pi * sec.r**3
    wzmin = wymin

    wxmin = ix / sec.r

    # TODO: This should be changed!!
    t = sec.r * 0.99
    dy = sec.r * 2
    di = dy - 2 * t
    sy = (dy**3 - di**3) / 12
    sz = sy
    shary = (2 * iz * t / sy) * sfy
    sharz = (2 * iy * t / sz) * sfz
    shceny = 0
    shcenz = 0
    cy = 0.0
    cz = 0.0

    return GeneralProperties(
        area=area,
        ix=ix,
        iy=iy,
        iz=iz,
        iyz=iyz,
        wxmin=wxmin,
        wymin=wymin,
        wzmin=wzmin,
        shary=shary,
        sharz=sharz,
        shceny=shceny,
        shcenz=shcenz,
        sy=sy,
        sz=sz,
        sfy=sfy,
        sfz=sfz,
        cy=cy,
        cz=cz,
        # parent=sec,
    )
