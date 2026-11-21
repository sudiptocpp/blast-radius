# Blast Radius selector

The selector uses dependencies measured on unmodified SQLGlot at `eb11aec4be8cb2122afe713a4816662c3f0fb2fa`. At runtime it reads each patch and selects tests from the supplied inventory. Missing one failing test adds 300 seconds per patch, so the selector widens its choice when the evidence is incomplete.

## How it works

1. [`build_coverage.py`](build_coverage.py) runs pytest against the pinned tree with the organizer's locked dependencies and `SKIP_INTEGRATION=0`. It records which tests execute each Python line, which test modules execute code during collection, and which files tests read directly. It also tracks shared fixture consumers, class attributes and dictionary keys. Pool jobs run inline with pickle round trips so their dependencies are attributed to the submitting test. Source reads made by the collector itself are excluded.
2. [`patches.py`](patches.py) reads the unified diff and reconstructs the edited Python in memory. Hunk positions and old source lines must match the pinned checkout. The runtime image contains the source and `coverage.pkl.zlib` and can run with a read-only filesystem and no network.
3. [`selector.py`](selector.py) matches recorded pytest IDs against the incoming `tests.json`. It anchors edits inside functions to old executable lines and enclosing statements. Insertions can change behavior for any test that entered the surrounding function. Direct fixture and source reads contribute their recorded consumers. For supported class assignments it uses observed attribute or key reads, including inherited values. Equivalent edits still select tests that read the source as data. Unsupported patches and uncertain import-time changes fall back to the full suite.
4. [`runner.py`](runner.py) verifies small candidate sets in forked processes with an in-memory patch overlay. Each fork retains failures and unverified candidates. A fixture with a single reader goes directly on the schedule. The Docker build records the files loaded by common imports so independent patches can share initialization. Patches touching those files run before the corresponding imports.
5. [`schedule.py`](schedule.py) tries up to eight worker counts. For each count it assigns tests in descending duration order to the least-loaded worker and keeps the schedule with the lowest combination of maximum load and per-worker overhead.

`coverage.pkl.zlib` comes entirely from the pinned baseline. The Docker build context excludes the sample files. Only the local scorer reads `sample/answers.json`.

## Reproduce and debug

From `blast-radius/` with Docker and GNU Make available, run `make sample` to build the image, run all 30 public patches under the container limits, and display the official local score. `output.jsonl` contains the chosen numeric test IDs per patch. Map those IDs to test names in `sample/tests.json` when investigating a miss.

Run `make coverage` after editing the coverage collector. It rebuilds the compressed index in the same Linux/Python environment used for scoring. Review the pytest summary for collection errors or unexpected test failures before using the new index. The normal image build reuses the checked-in index.

For local iteration with the locked dependencies, Python 3.13 and a checkout at the pinned commit, run the selector directly and write its stdout to `output.jsonl` as UTF-8. Score that output with the same scorer used by `make sample`.

```sh
python selector.py --data sample --checkout /path/to/pinned/sqlglot > output.jsonl
python score.py sample sample/answers.json output.jsonl
```

When a test is missing, start with its baseline coverage and determine whether the source ran during collection, was read as data, or ran in a subprocess. Fix the dependency signal or widen the general selection rule. Test IDs or labels from the sample answers should never go into the selector or index.

On the 30 public patches the original selector scored 492.798 simulated seconds and the optimized selector scored 94.603. Both covered every failing test. The leaderboard adds container CPU time. The coverage build observed 1,620 passing baseline tests and the organizer's single skipped test.

## Known gaps

- Baseline line coverage cannot observe paths first reached after a patch. Insertions use function-level coverage, and import-sensitive edits use wider selections.
- The runtime verifier uses unittest and an inline pool. It approximates pytest execution and process isolation. The CPU and time limits leave expensive or unfinished candidates on the schedule.
- The public sample provides a regression check. Hidden practice and final batches can contain different edits and fail in ways the sample does not exercise.
