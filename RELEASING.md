# Releasing

Same flow as py-svg-chart: bump2version manages versions, publishing a
GitHub Release triggers the PyPI upload.

## One-time setup

1. Create the GitHub repo and push.
2. Add the `PYPI_API_TOKEN` repository secret (PyPI → account settings →
   API tokens; scope it to the `macboost` project after the first upload).

## Each release

```sh
bump2version patch          # or minor / major — bumps pyproject.toml,
                            # __init__.py and Version.swift, commits,
                            # and tags vX.Y.Z (config: .bumpversion.cfg)
git push && git push --tags
gh release create vX.Y.Z --generate-notes    # or via the GitHub UI
```

Publishing the release runs `.github/workflows/python-publish.yml` on a
macOS arm64 runner: builds + tests the Swift core, builds the wheel
(`scripts/build_wheel.sh`, bundles the Metal dylib, tagged
`macosx_15_0_arm64`), attaches the wheel and the `macboost` CLI binary to
the release, and uploads the wheel to PyPI.

Note for the first release: verify the runner has Metal available (the
`swift test` step exercises the GPU); GitHub's arm64 macOS runners
generally do, but this is worth watching on run one.

Local dry run of the whole artifact build: `./scripts/build_wheel.sh`,
then `pip install python/dist/*.whl` in a clean venv.
