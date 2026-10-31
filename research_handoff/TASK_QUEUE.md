# Task queue

| ID | Priority | Dependencies | Status | Evidence / completion / next action |
|---|---|---|---|---|
| A1-A2 | P0 | none | IN_PROGRESS | Initial source audit done; original label independence/provenance unresolved; independent GT required. |
| A3 | P0 | independent target reference | BLOCKED_BY_HUMAN_REVIEW | 204 and 246 reviewed in logs and replay; anonymous candidate frames and human workflow in review/REVIEW_QUEUE.md. No human identity labels yet. |
| A4-A7 | P0 | reviewed labels | IN_PROGRESS | Existing metric audited; output-review converter and synthetic present-target wrong-output test added, 12 tests pass; requires independent human records to validate. |
| A8-A10 | P0 | data and reviewer | BLOCKED_BY_HUMAN_REVIEW | Identity protocol and review tool implemented; independent GT, ambiguity adjudication, and person/session split need humans/new source groups. Video 5 exploratory. |
| A11-A12 | P0 | A1-A10 | IN_PROGRESS | Gate A not passed. Decoder audit confirms 2699 packets/2698 frames; exact cause unknown. Monitor review completion and GT/split evidence. |
| B | P1 | Gate A | BLOCKED | ReID tuning/held-out comparison require reliable independent GT. |
| C-F | P1+ | relevant prior gates | NOT_STARTED | Safety contract, simulation, avoidance and hardware architecture. No real flight. |


## Human review checkpoint

- P0 implemented: local blind-first annotation, locked review, diagnostic queue, provenance, validated identity_screen ingestion and conflict checks.
- P0 HUMAN ACTION: independently identify the target and annotate priority frames before locking and reviewing output. UNKNOWN remains unresolved; no automatic identity labels.
- P0 NEXT: verify replay output against log and inspect human-reviewed session, resolve conflicts through documented independent adjudication. Geometry alone is not a false-lock label.
- Gate A remains blocked on independent human evidence, reviewed outcomes and group-based evaluation.
