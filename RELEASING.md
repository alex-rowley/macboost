# Releasing

bump2version manages versions; publishing a GitHub Release triggers the
PyPI upload.

## Each release

```sh
bump2version patch          # or minor / major — bumps pyproject.toml,
                            # __init__.py and Version.swift, commits,
                            # and tags vX.Y.Z (config: .bumpversion.cfg)
git push && git push --tags
gh release create vX.Y.Z --title vX.Y.Z --notes-file notes.md
```

Write the notes to a file; inline `--notes "$(cat <<'EOF' ...)"` breaks on
apostrophes. Patch for fixes, minor for new parameters or any change to
model behaviour or file formats.

Publishing the release runs `.github/workflows/python-publish.yml` on a
macOS arm64 runner: builds and tests the Swift core, builds the wheel
(`scripts/build_wheel.sh`, bundles the Metal dylib, tagged
`macosx_15_0_arm64`), attaches the wheel and the `macboost` CLI binary to
the release, and uploads the wheel to PyPI. The push to `main` runs the CI
workflow in parallel; both should be green before announcing.

## Before tagging

```sh
swift test                              # Swift behavioural suite
uv run scripts/test_python.py           # needs an arm64 Python (Homebrew /opt/homebrew)
./scripts/test_cli.sh
```

The default `python3` on an Intel-prefix Homebrew is x86_64 under Rosetta
and cannot train; `uv venv --python /opt/homebrew/bin/python3.13` gives an
arm64 environment.

## One-time setup

1. Create the GitHub repo and push.
2. Add the `PYPI_API_TOKEN` repository secret (PyPI → account settings →
   API tokens; scope it to the `macboost` project after the first upload).

Local dry run of the whole artifact build: `./scripts/build_wheel.sh`,
then `pip install python/dist/*.whl` in a clean venv.
