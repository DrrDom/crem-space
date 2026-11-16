"""The two spaces: grow (GrowSpace) and grow + make_cycle (GrowCycleSpace)."""


def open_space(slice_path, set_name, mode="grow", **kw):
    if mode == "grow":
        from cremspace.space.grow import GrowSpace
        return GrowSpace(slice_path, set_name, **kw)
    if mode == "cycle":
        from cremspace.space.cycle import GrowCycleSpace
        return GrowCycleSpace(slice_path, set_name, **kw)
    raise ValueError(f"unknown mode {mode!r} (grow or cycle)")
