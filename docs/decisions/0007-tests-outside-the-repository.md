# 0007. The test suite is not part of the repository

**Context.** The owner's position is that the tests were written for the authors, not for the world, and
that the repository should carry the product only.

**Decision.** `tests/` is ignored by git and there is no continuous integration. The suite (around 260
tests, including Atlas integration tests that run only with credentials) lives with the authors and is run
locally before every merge.

**Consequences.** A change is verified by running the suite on a developer machine, one file at a time
when memory is short. Reviewers cannot see test results on a pull request; the pull request description
states what was run. The early history tracked two test files before this decision, so switching between
commits from that period can delete the local copies: keep a backup outside the repository.
