# 0005. Connection tokens are opaque secrets in a ledger, never encrypted payloads

**Context.** The token a client holds for an open connection was an RSA-OAEP encryption of the user id,
the connection key and a nonce. RSA-OAEP with a 2048-bit key encrypts at most 190 bytes, and hosted session
ids pushed the payload past that limit, so hosted connections never worked.

**Decision.** `sql_agent/connections.py` issues `r1.` plus 32 random URL-safe bytes and keeps
`token -> (user, connection)` in a process-local ledger. Verification is a constant-time lookup; revocation
removes the entry. Nothing about the token is derivable from its text.

**Consequences.** Tokens are bound to the process that issued them, which matches the one-process
deployment and the connection handles that live in the same process. A multi-process deployment would move
the ledger to the cache, not back to encryption.
