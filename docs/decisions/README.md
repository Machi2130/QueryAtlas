# Architecture decision records

One file per decision that shapes the code and is not obvious from reading it: the context, the choice,
and what it rules out. Add a record when a future change could reasonably undo a decision without knowing
why it was made. Numbered in the order they were taken.

| # | Decision |
|---|---|
| [0001](0001-settings-object.md) | Every environment variable is read through one settings object |
| [0002](0002-connection-state.md) | One state object per open connection, shared by both products |
| [0003](0003-injectable-runtime.md) | Each product has an injectable runtime instead of module globals |
| [0004](0004-question-pipeline.md) | One question pipeline behind the three web question routes |
| [0005](0005-token-ledger.md) | Connection tokens are opaque secrets in a ledger, never encrypted payloads |
| [0006](0006-tenant-scope-by-rewriting.md) | The tenant filter is rewritten into every query by the server |
| [0007](0007-tests-outside-the-repository.md) | The test suite is not part of the repository |
| [0008](0008-one-process-one-model.md) | No process pool; the embedding model loads once per process |
