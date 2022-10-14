import ada
from ada.fem.formats.calculix.results.read_frd_file import read_from_frd_file
from ada.materials.metals import CarbonSteel, DnvGl16Mat


def gravity_step():
    step = ada.fem.StepImplicit("gravity", nl_geom=True, init_incr=50.0, total_time=100.0)
    step.add_load(ada.fem.LoadGravity("grav", -9.81 * 80))
    return step


def eigen_step():
    return ada.fem.StepEigen("gravity", 20)


def main():
    beam = ada.Beam(
        "MyBeam",
        (0, 0.5, 0.5),
        (3, 0.5, 0.5),
        "IPE400",
        ada.Material("S420", CarbonSteel("S420", plasticity_model=DnvGl16Mat(15e-3, "S355"))),
    )
    fem = beam.to_fem_obj(0.1, geom_repr="shell", use_hex=True)
    a = ada.Assembly("static_cantilever") / (ada.Part("P1", fem=fem) / beam)

    fix_set = fem.add_set(ada.fem.FemSet("bc_nodes", beam.bbox.sides.back(return_fem_nodes=True, fem=fem)))
    a.fem.add_bc(ada.fem.Bc("Fixed", fix_set, [1, 2, 3, 4, 5, 6]))

    is_static = False
    if is_static:
        prefix = "static"
        a.fem.add_step(gravity_step())
    else:
        prefix = "eigen"
        a.fem.add_step(eigen_step())

    rerun = True
    res_files = []
    res = a.to_fem(f"{prefix}_cantilever_abaqus", "abaqus", overwrite=rerun, execute=rerun)
    res_files.append(res.results_file_path)
    res = a.to_fem(f"{prefix}_cantilever_sesam", "sesam", overwrite=rerun, execute=rerun)
    res_files.append(res.results_file_path)
    res = a.to_fem(f"{prefix}_cantilever_code_aster", "code_aster", overwrite=rerun, execute=rerun)
    res_files.append(res.results_file_path)
    res = a.to_fem(f"{prefix}_cantilever_calculix", "calculix", overwrite=rerun, execute=rerun)
    res_files.append(res.results_file_path)

    for resf in res_files:
        mesh = read_from_frd_file(resf)
        mesh.write(resf.with_suffix(".vtu"))


if __name__ == "__main__":
    main()
