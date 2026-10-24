# Ground-Truth Draft Notes – videos/5.mp4

**Method:** AI-drafted, not yet verified  
**Date:** 2026-10-21  
**Status:** AI-drafted, not yet verified

---

## Method Summary

Labels were derived **exclusively** from the raw video `videos/5.mp4` by visual inspection of contact sheets (one tile every 30 frames) plus fine-grained boundary strips (every 1-3 frames near transitions). No tracker output, log files, or existing ground_truth/5.csv were consulted.

**Target identification criteria used:**
- Denim jacket (blue, distinct shade) over black turtleneck
- Dark backpack (black/very dark, moderately large)
- Often holding a clear bottle with orange/amber drink and green cap
- Visible from front, back, or side

**Distractor (NOT the target):**
- Man in grey hoodie + glasses + black backpack + similar bottle (appears F904–~F1280)

---

## Unsure Samples (U labels in samples CSV)

| Frame | Reason | Resolution after review |
|-------|--------|------------------------|
| F2160 | Initially marked U due to umbrella entering frame. On closer review of the full-size frame: denim jacket + dark backpack are clearly visible from the back above the umbrella. | **Reclassified: V** (falls in V interval 1743–2161) |
| F2190 | Umbrella occlusion. Nike-cap glasses man (not target) visible. Target body completely absent. | **Reclassified: N** (falls in N interval 2162–2396) |

Neither frame was left as U in the final 5_ai_draft.csv. The boundary (V→N at F2162) is consistent with both reclassifications.

---

## Boundary Decisions

### Boundary 1: V→N at F904
- F900–F903: Target clearly visible (denim jacket, black turtleneck, front view, smiling)
- F904: Hard camera cut; grey hoodie + glasses man fills frame
- Confirmed by bnd1_F900_F911 strip (every frame F900-F911)
- **Edge case: none** — cut is abrupt

### Boundary 2: N→V at F1743
- F1740–F1742: Two older men walking past beer shop, no target
- F1743: Denim jacket + black turtleneck man appears (front, drinking bottle)
- Confirmed by bnd3_F1740_F1751 strip (every frame F1740-F1751)
- **Edge case: F1740-F1742** — no denim jacket visible

### Boundary 3: V→N at F2162
- F2090–F2161: Target walking away from camera (back view), denim jacket and dark backpack clearly identifiable. The umbrella from another person starts entering from the left at F2155 but does not yet block the jacket.
- F2162: Large white umbrella sweeps to centre-frame, target body no longer clearly identifiable (only a glimpse of the backpack top on right edge)
- F2172+: Camera focus switches to glasses man with Nike cap and then woman in white coat; target gone
- Confirmed by T3_V_to_N_at_F2162 (4-panel) and bndC_fine_F2108_F2165
- **Edge cases: F2159–F2165** — umbrella occlusion window; F2161 is still V by the back-view rule, F2162 is the first genuinely ambiguous frame
- Note: F2090–F2161 are "back" view. Even though the face is not seen, the denim jacket and dark backpack are unambiguously recognisable from behind.

### Boundary 4: N→V at F2397
- F2380–F2396: Camera follows a woman with long brown hair; denim jacket person visible in background but occluded by woman's body/hair (not clearly identifiable)
- F2397: Denim jacket + dark backpack in full back view, walking away, clearly identifiable
- Confirmed by bndD_fine_F2380_F2410 (every frame F2380-F2410)
- **Edge cases: F2380–F2396** — partial glimpse of denim-coloured jacket behind woman but not enough to confirm target

### Boundary 5: V→N at F2510
- F2505–F2509: Target in front view at "Since 1900" shop, clearly visible (denim jacket, turtleneck)
- F2510: Screen suddenly goes dark/split with phone screen (YouTube UI) overlaid top half
- F2511+: Full YouTube-style phone screen overlay, no target body visible
- Confirmed by bnd6_F2505_F2526 (every frame F2505-F2526)
- **Edge case: none** — cut is abrupt

### Boundary 6: N→V at F2560
- F2511–F2559: Phone/YouTube screen overlay, or brief dark frame; no target visible
- F2560: Target's back visible (denim jacket, dark backpack) walking past Tsingtao brewery
- Confirmed by bnd7_F2555_F2595 (every frame F2555-F2595)
- **Edge case: F2555–F2559** — partial screen transition; one frame may show the brewery before the target is fully in frame

---

## Final Interval Table

| start_frame | end_frame | target_visible | view  | note |
|-------------|-----------|----------------|-------|------|
| 0           | 903       | 1              | front | Clear front view throughout; bottle prominent |
| 904         | 1742      | 0              | none  | Distractor (grey hoodie) F904-~F1280; crowd F1280-F1742 |
| 1743        | 2161      | 1              | front | Front view F1743-~F2089; back view F2090-F2161 |
| 2162        | 2396      | 0              | none  | Umbrella block then camera follows different people |
| 2397        | 2509      | 1              | back  | Back view; front view briefly around F2460-F2509 |
| 2510        | 2559      | 0              | none  | Phone screen overlay |
| 2560        | 2697      | 1              | back  | Back view at Tsingtao brewery; screen overlay visible but target body visible |

Total frames: 2698 (0–2697)  
Visible frames: 904 + 419 + 113 + 138 = **1574**  
Not visible frames: 839 + 235 + 50 = **1124**
