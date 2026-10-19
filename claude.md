## Testing policy

Test only what the change touches. Do not run tests "to be safe".

- During work: run only the tests for the files you changed
  (e.g. `pytest tests/test_<module>.py`, or `-k <name>`). No full suite.
- Full suite (`pytest`, `npm test`, `tsc`): run ONCE, at the end of a task that
  changed shared code (detection pipeline, matching, build model, API). Skip it
  for UI text, styling, docs or config-only changes.
- Never re-run a passing test. If a test fails, fix it and re-run only that test.
- A flaky test: report it, don't loop on it.
- No live external calls in tests (Gemini, Brickognize, Brickset, Rebrickable).
  Use recorded/mocked responses. Live calls only when I explicitly ask.
- No headless browser runs, screenshots or performance measurements unless I ask
  or the task is about the UI behaviour itself.
- Don't run the detection pipeline over all sample photos unless the task
  changes detection. Then run it once and report the metrics table.
- When unsure whether a test run is needed, skip it and tell me which test
  you'd run.

## Reporting

Keep reports short: what changed, which tests you ran, anything I must decide.## External APIs

- Never run benchmarks, comparisons or batch runs against Gemini, Brickognize or
  Brickset without asking me first. State the expected number of calls and wait.
