## 0.0.4 (2026-01-20)

* Use `conda>=25.5.0`, `py-rattler>=0.20`, `python>=3.10` by @jaimergp in #23
* Do not report "not found" packages as missing if already installed by @jaimergp in #25
* Prettify output, tidy up comments, add docstrings by @jaimergp in #26
* Update some deprecated imports by @jaimergp in #31
* Fix Python version changes fast-tracking by @jaimergp in #27
* CI: Update and pin GHA actions versions by @jaimergp in #24
* CI: Fix CI matrix for macOS by @jaimergp in #31
* CI: Update tests workflow from conda/conda by @jaimergp in #33
* CI: Bump setup-miniconda and remove xonsh workarounds by @jaimergp in #34
* CI: Use macos-15-intel by @jaimergp in #17

## 0.0.3 (2025-05-25)

* Use new APIs in `SparseRepoData` (`.record_count()` and `PackageFormatSelection` enum) by @jaimergp in #12
* Implements `.search()` with `SparseRepoData.load_matching_records()` by @jaimergp in #13
* Use `.record_count()` from py-rattler 0.13.1 by @jaimergp in #14
* Add conda-build recipe, fix version logic by @jaimergp in #15

## 0.0.2 (2025-05-21)

* Convert records without temporary JSON file by @jaimergp in #11

## 0.0.1 (2025-05-20)

**Prototype release**. Not meant for production.

Passes most of the test suite, except for:

- `defaults::package` does not work
- Some channel priority differences when mixing channels
