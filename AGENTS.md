# Repository notes

This repository is a collection of Pluto notebooks (`.jl` sources with embedded
`PLUTO_PROJECT_TOML_CONTENTS` / `PLUTO_MANIFEST_TOML_CONTENTS`) plus checked-in
generated `.html` exports.

## Verifying notebooks

There is no package test suite; the meaningful check is executing each notebook
in Pluto and regenerating its HTML export:

- Use `Pluto.SessionActions.open(session, path; run_async=false)` with
  `session.options.evaluation.workspace_use_distributed = true`. Non-distributed
  evaluation does not activate the notebook's embedded package environment, so
  `using Package` cells fail.
- Check `cell.errored` after the run; `Pluto.generate_html(nb)` writes the
  export with run outputs baked in.
- To update embedded environments without a noisy whole-file rewrite, call
  `Pluto.update_nbpkg(session, nb; save=false)`, read the resolved
  Project/Manifest from the notebook's scratch env dir, and splice only those
  TOML blocks back into the original file.

## Pluto gotchas seen here

- Cell execution order is a topological sort, not file/cell-order. Indexing
  syntax `a[i]` does not create a dependency edge to cells that define
  `Base.getindex` methods (explicit `getindex(a, i)` calls do). Keep
  `Base.getindex`/`Base.size`/`Base.eltype` method definitions in the same cell
  as the type they extend (wrapped in `begin ... end`), so the struct reference
  in test cells forces correct ordering.
- Pluto requires a single expression per cell; multiple top-level forms need a
  `begin ... end` wrapper.
