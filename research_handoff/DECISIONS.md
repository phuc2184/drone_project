# Decisions

2026-10-29: Preserve existing modified and untracked files, original ground truth, videos, logs and weights. Use separate research_handoff artifacts and tests. No GitHub writes.

2026-10-29: Treat low IoU as a review candidate, not proven identity error; overlap geometry cannot establish identity. Separate TRACKING and INITIALIZING and treat visibility uncertainty explicitly. Implement reviewed identity metrics but do not fabricate reviews.

2026-10-29: Video 5 is exploratory because outputs were already inspected. Do not retroactively label it held-out. Split new independent videos by person/session when available; A gate remains unmet. Alternative frame-level random split risks temporal and identity leakage.

2026-10-29 continuation: Manual review is two-stage: scene on raw frame before anonymous candidate boxes; output role, track ID and ReID score hidden. Candidate generation conditions review on detector proposals, so independent box annotation remains necessary. NOT_VISIBLE never maps to confirmed absence; LOW or missing review becomes uncertain in conversion. Alternatives that expose tracker ID or infer identity from IoU create leakage and were rejected. Source video and manifest hashes accompany each review; leave user-generated reviews unmodified.
