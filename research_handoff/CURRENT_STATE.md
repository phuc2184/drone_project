# Current state — 2026-10-29

Phase A IN_PROGRESS; Gate A NOT PASSED. HEAD eb3eb1f3623ed7cc2dc1c7f0821edaece667d3e9; original Git state: M ROADMAP.md, STATUS.md, roadmap_data.json; untracked EYE_LABEL_REPORT.md, analyze_eye_labels.py. Preserved. New research_handoff/ and tests/ remain local and untracked; no GitHub writes.

VERIFIED: original evaluator misses wrong identity when target present. New geometry screening and manually reviewed identity metrics distinguish present-target wrong-output, uncertainty, and INITIALIZING/TRACKING. Blind candidate workflow, post-review join, annotation protocol and diagnostic frame list in `review/`; 12 tests pass. Mock GUI testing only; no human review or ground-truth labels have been invented.

MEASUREMENT: Replay for frames 0–2695 matches state/target ID/valid on 2696/2696 log rows. 20 candidate frames saved; replayed 204 and 246 are present in selected candidate frames; selected frame IDs and limitations in review/REVIEW_QUEUE.md. Seven output/hand-label IoUs round to the prior measurements. FFmpeg independent audit metadata 2699 video packets, 2698 decoded frames, normal EOF and no decode errors; reason is UNKNOWN.

BLOCKED_BY_HUMAN_REVIEW: review identities with independently established target reference, especially 204/246; do not infer identity from ID, score or IoU. Protocol `review/ANNOTATION_PROTOCOL.md`; review tool requires local GUI and a new output JSONL. Videos/person/session groups needed for independent GT and held-out split; video 5 is exploratory. Continue with real human review, inter-reviewer adjudication and leakage-free new data; consult TASK_QUEUE.md, PHASE_A_REPORT.md and EXPERIMENT_LOG.md.


## Human identity review — 2026-10-29

Implemented local HTTP review `review/identity_review.py`; raw frames use sequential zero-based decoding. New JSONL sessions contain versioned human annotations and explicit annotation lock before tracking review. `identity_screen.py --review-session` validates provenance, contradictions and UNKNOWN; `TARGET_NOT_VISIBLE` is not absence. No actual identity labels have been supplied. Review queue is diagnostic and nonrepresentative; historical prior knowledge and frame selection prevent claiming held-out blinding. Log contains no boxes: output box comparison requires a separately generated checked replay manifest. Browser visualization unavailable in this environment; HTTP smoke tests and unit tests provide limited validation. Gate A remains NOT PASSED.

MEASUREMENT: replay `review/diagnostic_replay.json` contains frames 0, 60, 188, 904, 905, 1061, 1062, 1066, 1140, 1141, 1145. `extract_candidates.py` checked state/target ID/valid through frame 1145, but log lacks historical boxes so replay box equality cannot be verified. HTTP smoke: root/info/frame 1061/video byte range succeeded and comparison before lock returned 400. Browser host unavailable; visual interaction not tested. No human identity review performed.
