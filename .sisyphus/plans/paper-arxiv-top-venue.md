# Plan: Strengthen the paper into a 10/10 arXiv preprint

## TL;DR

> **Quick Summary**: Turn the current manuscript in `paper/` into a submission-safe, high-quality arXiv preprint by sharpening contribution identity, tightening claims to evidence, improving results storytelling, strengthening reproducibility packaging, and performing a final arXiv dry run.
>
> **Deliverables**:
> - A cleaner, sharper manuscript in `paper/main.tex` and `paper/sections/*.tex`
> - Improved figures/tables/captions and narrative hierarchy
> - A verified arXiv submission package rooted at `paper/`
> - A stronger reproducibility story aligned with the repository `README.md`
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 3 implementation waves + final verification
> **Critical Path**: T1 Contribution identity → T7 Intro/abstract rewrite → T10 Results restructuring → T15 full compile/package validation → T16 arXiv dry run

---

## Context

### Original Request
User has an existing LaTeX paper under `paper/`, wants it improved into a “10/10” arXiv preprint now, and wants that version to be easy to reformat later into a top-venue submission.

### Interview Summary
**Key Discussions**:
- No specific venue is targeted yet.
- The immediate goal is a strong arXiv preprint rather than venue-template tailoring.
- User wants a full plan to improve all observed weaknesses.
- The most important current weaknesses are evidence robustness, contribution identity, selector justification, intro punch, and results storytelling.

**Research Findings**:
- `paper/README.md` already contains an arXiv packaging checklist and identifies `main.tex` as the top-level source.
- `paper/main.tex` is already an arXiv-friendly `article` setup using `natbib` + `plainnat`.
- The manuscript already has strong structural coverage: introduction, related work, design, setup, results, discussion, limitations, conclusion, appendix reproducibility.
- The current manuscript appears strongest as a systems/ML-systems preprint, but currently mixes artifact, measurement, and policy-paper identities.

### Metis Review
**Identified Gaps** (addressed in this plan):
- Missing explicit guardrails around claim inflation and scope creep → Added must-have / must-not-have rules and a separate evidence/claim audit wave.
- No explicit acceptance criteria for “arXiv-ready” versus “strengthened preprint” → Added task-level criteria and a final arXiv dry-run gate.
- Risk of over-expanding into a new research project → Locked plan to manuscript strengthening first, with new experiments only when they directly de-risk current claims.
- Missing distinction between must-fix submission blockers and quality upgrades → Execution waves separate foundation, strengthening, and packaging.

---

## Work Objectives

### Core Objective
Produce a submission-safe arXiv paper that reads like a serious top-venue candidate: crisp contribution identity, disciplined claims, reviewer-efficient results presentation, reproducibility clarity, and a clean final package.

### Concrete Deliverables
- Updated paper source under `paper/main.tex` and `paper/sections/*.tex`
- Improved figures/tables/captions under `paper/figures/` and/or inline TikZ sections
- Synchronized reproducibility instructions across `paper/README.md`, `paper/sections/appendix_reproducibility.tex`, and repository `README.md`
- Verified arXiv package containing `main.tex`, `00README.json`, `sections/`, `bib/references.bib`, and any required `figures/` assets

### Definition of Done
- [x] `pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex` runs successfully from `paper/`
- [x] The generated PDF presents one unmistakable main contribution and no unsupported claim survives
- [x] arXiv-required files are packaged correctly with no generated junk at archive root
- [x] The final manuscript clearly separates proven claims, bounded claims, and future-work claims

### Must Have
- A singular, memorable paper identity
- Claims that precisely match the measured evidence
- Stronger introduction, abstract, and conclusion alignment
- Results that are visually and narratively easier to follow than the current matrix-heavy presentation
- Reproducibility instructions that are internally consistent across paper and repo

### Must NOT Have (Guardrails)
- No new claims without supporting evidence
- No conversion of this effort into a wholly new research project
- No venue-specific rewrite that harms general arXiv readability
- No paper-wide rewrite from scratch unless a section is irredeemable
- No breaking arXiv-readiness while chasing polish
- No burying the main contribution under over-defensive caveats

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (LaTeX build process and package structure exist)
- **Automated tests**: Tests-after
- **Framework**: `pdflatex`/`bibtex` build, file/package checks, content audits, grep/read validation

### QA Policy
Every task must include agent-executed QA scenarios. Evidence goes in `.sisyphus/evidence/`.

- **Manuscript checks**: Use Read/Grep to confirm required narrative/claim changes landed.
- **Build checks**: Use Bash in `paper/` to compile and inspect logs.
- **Package checks**: Use Bash to create/list the arXiv archive root and confirm allowed files only.
- **Cross-file consistency**: Use Read/Grep to compare wording across abstract, intro, results, discussion, conclusion, README, and appendix.

---

## Execution Strategy

### Parallel Execution Waves

Wave 1 (Start Immediately - foundation + audit):
├── Task 1: Lock the paper’s main contribution and paper identity [writing]
├── Task 2: Build a claim-to-evidence ledger and soften unsupported claims [unspecified-high]
├── Task 3: Audit arXiv packaging/build assumptions [quick]
├── Task 4: Inventory figures, tables, and caption weaknesses [writing]
├── Task 5: Audit related-work positioning and novelty contrast [writing]
└── Task 6: Audit reproducibility and repo-paper alignment [unspecified-high]

Wave 2 (After Wave 1 - core manuscript strengthening):
├── Task 7: Rewrite title, abstract, and introduction around one thesis [writing]
├── Task 8: Rewrite related work and narrative scope for sharper positioning [writing]
├── Task 9: Rewrite experimental setup for reviewer trust and evidence clarity [writing]
├── Task 10: Restructure results into a reviewer-efficient story [writing]
├── Task 11: Make selector/default-policy justification explicit and bounded [unspecified-high]
└── Task 12: Tighten discussion, limitations, and conclusion [writing]

Wave 3 (After Wave 2 - integration + polish + submission prep):
├── Task 13: Align cross-section terminology, claims, and section transitions [writing]
├── Task 14: Improve figures/tables/captions and final manuscript readability [writing]
├── Task 15: Run full LaTeX compile pass and fix format/bib/ref issues [quick]
├── Task 16: Produce arXiv package and perform dry-run validation [quick]
└── Task 17: Final preprint polish pass for public release quality [writing]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Manuscript quality and build review (unspecified-high)
├── Task F3: Real QA of compile + package + PDF checks (unspecified-high)
└── Task F4: Scope fidelity and claim-discipline check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T7 → T10 → T15 → T16 → F1-F4 → user okay
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 6

### Dependency Matrix

- **T1**: Blocked By: None | Blocks: T7, T8, T12, T13
- **T2**: Blocked By: None | Blocks: T7, T9, T10, T11, T12, T13
- **T3**: Blocked By: None | Blocks: T15, T16, T17
- **T4**: Blocked By: None | Blocks: T10, T14, T17
- **T5**: Blocked By: None | Blocks: T8, T12
- **T6**: Blocked By: None | Blocks: T9, T13, T16
- **T7**: Blocked By: T1, T2 | Blocks: T13, T17
- **T8**: Blocked By: T1, T5 | Blocks: T13, T17
- **T9**: Blocked By: T2, T6 | Blocks: T13, T15
- **T10**: Blocked By: T2, T4 | Blocks: T14, T17
- **T11**: Blocked By: T2 | Blocks: T12, T13, T17
- **T12**: Blocked By: T1, T2, T5, T11 | Blocks: T13, T17
- **T13**: Blocked By: T1, T2, T6, T7, T8, T9, T11, T12 | Blocks: T15, T17
- **T14**: Blocked By: T4, T10 | Blocks: T15, T17
- **T15**: Blocked By: T3, T9, T13, T14 | Blocks: T16, T17
- **T16**: Blocked By: T3, T6, T15 | Blocks: F1-F4
- **T17**: Blocked By: T3, T4, T7, T8, T10, T11, T12, T13, T14, T15 | Blocks: F1-F4

### Agent Dispatch Summary

- **Wave 1**: 6 agents — T1 `writing`, T2 `unspecified-high`, T3 `quick`, T4 `writing`, T5 `writing`, T6 `unspecified-high`
- **Wave 2**: 6 agents — T7 `writing`, T8 `writing`, T9 `writing`, T10 `writing`, T11 `unspecified-high`, T12 `writing`
- **Wave 3**: 5 agents — T13 `writing`, T14 `writing`, T15 `quick`, T16 `quick`, T17 `writing`
- **FINAL**: 4 agents — F1 `oracle`, F2 `unspecified-high`, F3 `unspecified-high`, F4 `deep`

---

## TODOs

- [x] T1. Lock the paper’s main contribution and paper identity

  **What to do**:
  - Decide and document the dominant identity of the manuscript: measurement-driven systems paper with operational policy contribution, not a diffuse artifact/benchmark/everything paper.
  - Draft a one-sentence thesis, a one-sentence novelty claim, and a one-sentence operational takeaway that all later sections must align to.
  - Identify what should be demoted to supporting contribution status (e.g. framework implementation details that are not the central claim).

  **Must NOT do**:
  - Do not invent a new contribution unsupported by the current work.
  - Do not let the manuscript keep multiple equal-weight identities.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: This is narrative framing and contribution-definition work.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Keeps the work grounded in an existing LaTeX paper project.
    - `writing-clearly-and-concisely`: Helps compress the paper identity into strong, memorable prose.
  - **Skills Evaluated but Omitted**:
    - `academic-researcher`: Not needed yet because this task is internal framing, not literature expansion.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T2-T6)
  - **Blocks**: T7, T8, T12, T13
  - **Blocked By**: None

  **References**:
  - `paper/main.tex:45-54` - Abstract currently states the paper’s outward-facing thesis; this is the first place contribution identity must become singular.
  - `paper/sections/introduction.tex:10-30` - Current intro already states gap, goals, and contributions, but mixes several identities; use this as the main rewrite anchor.
  - `paper/sections/conclusion.tex:4-30` - Conclusion shows how the paper currently “lands”; the final identity must align here too.
  - `paper/sections/narrative_scope.tex:4-26` - Current policy framing around scale/layout/adaptation should inform the paper’s true central claim.

  **Acceptance Criteria**:
  - [ ] A single dominant paper identity is documented in the manuscript notes or plan execution output.
  - [ ] Abstract, intro contributions, and conclusion all express the same central thesis.
  - [ ] At least one previously diffuse or secondary story is demoted or reframed.

  **QA Scenarios**:
  ```
  Scenario: Identity alignment exists across core sections
    Tool: Read + Grep
    Preconditions: T1 edits complete
    Steps:
      1. Read abstract, introduction contribution paragraph, and conclusion summary.
      2. Grep for thesis keywords across `paper/main.tex` and `paper/sections/*.tex`.
      3. Assert the same main contribution appears in all three places with no contradictory framing.
    Expected Result: One dominant thesis is visible and repeated consistently.
    Failure Indicators: Different sections frame the paper as different kinds of contribution.
    Evidence: .sisyphus/evidence/task-t1-identity-alignment.txt

  Scenario: Diffuse identity removed
    Tool: Read
    Preconditions: T1 edits complete
    Steps:
      1. Inspect the contribution list in `paper/sections/introduction.tex`.
      2. Verify there is a clear primary contribution and supporting sub-points.
    Expected Result: Contribution hierarchy is explicit.
    Evidence: .sisyphus/evidence/task-t1-identity-hierarchy.txt
  ```

  **Commit**: NO

- [x] T2. Build a claim-to-evidence ledger and soften unsupported claims

  **What to do**:
  - Enumerate every strong claim in abstract, introduction, results, discussion, and conclusion.
  - Map each claim to evidence already present in tables, figures, setup, or appendix.
  - Soften, bound, or remove any claim whose evidence is weaker than the wording suggests.
  - Distinguish empirical claim, mechanistic interpretation, and operational recommendation.

  **Must NOT do**:
  - Do not keep broad generalizations unsupported by current measured scope.
  - Do not weaken true strong claims just because they are bold; only align them to evidence.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: This is a high-judgment audit spanning many sections.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Helps preserve paper structure while auditing claims.
    - `writing-clearly-and-concisely`: Useful for converting vague claims into precise ones.
  - **Skills Evaluated but Omitted**:
    - `deep-research`: Not required unless new external evidence is needed.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T3-T6)
  - **Blocks**: T7, T9, T10, T11, T12, T13
  - **Blocked By**: None

  **References**:
  - `paper/main.tex:47-54` - Abstract contains the highest-risk summary claims.
  - `paper/sections/introduction.tex:12-25` - Intro gap statement and contributions need evidence-backed wording.
  - `paper/sections/results.tex:11-236` - Main evidence base for performance and policy claims.
  - `paper/sections/discussion.tex:18-40` - Interpretation and policy claims must stay bounded.
  - `paper/sections/limitations.tex:18-34` - Existing caveats define the acceptable claim envelope.

  **Acceptance Criteria**:
  - [x] Every strong claim in abstract/introduction/conclusion maps to a cited table, figure, or bounded limitation.
  - [ ] Unsupported universal language (“always”, “general”, “optimal”) is removed unless proven.
  - [ ] The manuscript explicitly separates measured findings from interpretations.

  **QA Scenarios**:
  ```
  Scenario: All main claims have evidence anchors
    Tool: Read
    Preconditions: T2 edits complete
    Steps:
      1. Read abstract, intro contribution list, conclusion, and results tables.
      2. Check each main claim against a table/figure/limitations reference.
      3. Record any claim without an evidence anchor.
    Expected Result: Zero unanchored headline claims remain.
    Failure Indicators: A claim in abstract/conclusion has no supporting evidence location.
    Evidence: .sisyphus/evidence/task-t2-claim-ledger.txt

  Scenario: Overclaim language removed
    Tool: Grep
    Preconditions: T2 edits complete
    Steps:
      1. Search for high-risk universal terms across paper sources.
      2. Inspect each match manually.
    Expected Result: Any remaining universal wording is justified or bounded.
    Evidence: .sisyphus/evidence/task-t2-overclaim-scan.txt
  ```

  **Commit**: NO

- [x] T3. Audit arXiv packaging/build assumptions

  **What to do**:
  - Validate that the paper can be packaged for arXiv exactly as documented.
  - Confirm the root archive contents, processor choice, `00README.json`, abstract text file, and omission of generated outputs.
  - Identify any hidden local dependency (missing `.bst`, missing figure asset, local macro assumption, unsupported package usage).

  **Must NOT do**:
  - Do not assume arXiv will infer local environment quirks correctly.
  - Do not leave packaging validation only for the very end.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: This is a bounded packaging/build audit.
  - **Skills**: [`latex-paper-en`]
    - `latex-paper-en`: Covers compile/build/formatting concerns for existing LaTeX projects.
  - **Skills Evaluated but Omitted**:
    - `writing-clearly-and-concisely`: Packaging validation is procedural, not prose-heavy.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1, T2, T4-T6)
  - **Blocks**: T15, T16, T17
  - **Blocked By**: None

  **References**:
  - `paper/README.md:20-40` - Existing arXiv packaging checklist and expected root contents.
  - `paper/00README.json` - Declares top-level source and processor expectations.
  - `paper/arxiv_abstract_plain.txt` - Plain abstract to sync with manuscript.
  - `paper/main.tex:1-24` - Package/class assumptions that must remain arXiv-safe.

  **Acceptance Criteria**:
  - [ ] Required arXiv files are identified and sufficient.
  - [ ] Any unsupported local dependency is discovered before final packaging.
  - [ ] Build/package instructions in `paper/README.md` remain correct after later edits.

  **QA Scenarios**:
  ```
  Scenario: Required arXiv files are complete
    Tool: Read + Bash
    Preconditions: T3 audit complete
    Steps:
      1. Read `paper/README.md` packaging section.
      2. List current `paper/` contents.
      3. Assert that required source files exist and generated junk is identified for exclusion.
    Expected Result: A complete include/exclude manifest exists.
    Failure Indicators: Missing required file or ambiguous package root layout.
    Evidence: .sisyphus/evidence/task-t3-package-manifest.txt

  Scenario: Abstract consistency check
    Tool: Read
    Preconditions: T3 audit complete
    Steps:
      1. Read `paper/arxiv_abstract_plain.txt` and the LaTeX abstract.
      2. Compare for meaning-level consistency.
    Expected Result: arXiv web-form abstract matches the manuscript abstract.
    Evidence: .sisyphus/evidence/task-t3-abstract-consistency.txt
  ```

  **Commit**: NO

- [x] T4. Inventory figures, tables, and caption weaknesses

  **What to do**:
  - Review every figure and table for story value, readability, caption quality, and redundancy.
  - Identify missing “summary” visuals that would reduce reviewer effort.
  - Flag tables that should become plots, plots that need annotation, and captions that fail to state the takeaway.

  **Must NOT do**:
  - Do not keep every existing visual if some dilute the main story.
  - Do not rely on body text alone to explain what a figure/table means.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: This is narrative and communication design for technical visuals.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Handles figure/caption review in LaTeX context.
    - `writing-clearly-and-concisely`: Helps craft take-away-first captions.
  - **Skills Evaluated but Omitted**:
    - `chart-visualization`: This task audits scientific storytelling, not generating dashboard-style charts from scratch.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1-T3, T5-T6)
  - **Blocks**: T10, T14, T17
  - **Blocked By**: None

  **References**:
  - `paper/sections/results.tex:23-127` - Existing main SafeTensors and ServerlessLLM tables/plots.
  - `paper/sections/results.tex:145-224` - Python and vLLM tables needing hierarchy and caption clarity.
  - `paper/figures/` - Any external figure assets used by the manuscript.

  **Acceptance Criteria**:
  - [ ] Every retained figure/table has a unique narrative purpose.
  - [ ] Every caption states the intended takeaway, not just what is displayed.
  - [ ] At least one summary visual opportunity is identified for the results story if missing.

  **QA Scenarios**:
  ```
  Scenario: Visual inventory completed
    Tool: Read + Glob
    Preconditions: T4 audit complete
    Steps:
      1. Enumerate all figures and tables used in results and related sections.
      2. Record each item’s purpose and whether it is retained, revised, merged, or removed.
    Expected Result: No visual remains without a role.
    Evidence: .sisyphus/evidence/task-t4-visual-inventory.txt

  Scenario: Captions carry takeaways
    Tool: Read
    Preconditions: T4 edits complete
    Steps:
      1. Read each caption.
      2. Assert that each caption contains a takeaway or interpretation sentence.
    Expected Result: Reader can understand the point of each visual from caption + axes/table labels.
    Evidence: .sisyphus/evidence/task-t4-caption-audit.txt
  ```

  **Commit**: NO

- [x] T5. Audit related-work positioning and novelty contrast

  **What to do**:
  - Sharpen the “what prior work does not settle” distinction.
  - Ensure the manuscript answers “why this paper exists” in one reviewer-efficient paragraph.
  - Reduce broad literature summary where it does not directly strengthen novelty contrast.
  - Make the relationship to SafeTensors, serverless cold-start work, database `io_uring` studies, and vLLM unmistakably complementary rather than overlapping.

  **Must NOT do**:
  - Do not bloat related work into an exhaustive survey.
  - Do not overstate novelty by pretending adjacent work does not exist.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: This is positioning and synthesis prose.
  - **Skills**: [`latex-paper-en`, `academic-researcher`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Keeps changes grounded in the current LaTeX paper.
    - `academic-researcher`: Useful for checking that novelty contrast is intellectually honest.
    - `writing-clearly-and-concisely`: Helps compress the “gap” argument.
  - **Skills Evaluated but Omitted**:
    - `deep-research`: Too heavy unless major new literature review is required.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1-T4, T6)
  - **Blocks**: T8, T12
  - **Blocked By**: None

  **References**:
  - `paper/sections/related_work.tex:6-92` - Existing literature positioning and synthesis.
  - `paper/sections/introduction.tex:12-18` - Intro’s current statement of what prior work does not settle.
  - `paper/sections/narrative_scope.tex:4-26` - Core thesis that related work should point toward.

  **Acceptance Criteria**:
  - [ ] Related work ends with a crisp, non-defensive novelty contrast.
  - [ ] A reviewer can answer “what exact gap does this paper fill?” from one short passage.
  - [ ] The paper no longer sounds like a generic benchmark of adjacent topics.

  **QA Scenarios**:
  ```
  Scenario: Gap statement is explicit
    Tool: Read
    Preconditions: T5 edits complete
    Steps:
      1. Read the related work conclusion/positioning paragraph and intro gap paragraph.
      2. Verify they answer the same exact gap question.
    Expected Result: Both sections state one crisp gap with no contradiction.
    Evidence: .sisyphus/evidence/task-t5-gap-statement.txt

  Scenario: Related work is not bloated
    Tool: Read
    Preconditions: T5 edits complete
    Steps:
      1. Inspect related work subsection endings.
      2. Confirm each subsection contributes to novelty positioning, not literature dumping.
    Expected Result: Each subsection has a clear relevance to the paper’s thesis.
    Evidence: .sisyphus/evidence/task-t5-related-work-focus.txt
  ```

  **Commit**: NO

- [x] T6. Audit reproducibility and repo-paper alignment

  **What to do**:
  - Compare the paper’s reproducibility appendix and setup section against the repository README and scripts.
  - Align commands, paths, model IDs, output directories, and environment recording guidance.
  - Remove any stale or contradictory replication instructions.
  - Ensure the preprint’s reproducibility story is usable by an external reader.

  **Must NOT do**:
  - Do not claim “fully reproducible” if replication is only directional or hardware-sensitive.
  - Do not leave paper and repo describing different workflows.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Cross-document consistency audit with technical detail.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Useful for appendix/reproducibility section review.
    - `writing-clearly-and-concisely`: Helps turn complex replication instructions into precise text.
  - **Skills Evaluated but Omitted**:
    - `academic-researcher`: Not necessary because this is repo/paper consistency, not external literature.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with T1-T5)
  - **Blocks**: T9, T13, T16
  - **Blocked By**: None

  **References**:
  - `paper/README.md:1-40` - Paper-local build and arXiv packaging instructions.
  - `README.md` - Repository-wide reproduction commands and environment capture.
  - `paper/sections/experimental_setup.tex:22-135` - Setup and replication framing in the manuscript.
  - `paper/sections/appendix_reproducibility.tex` - Detailed appendix source of truth for replication claims.

  **Acceptance Criteria**:
  - [ ] Paper and repository no longer disagree on commands, paths, or output locations.
  - [ ] Reproducibility wording correctly distinguishes exact timing vs ordering/regime replication.
  - [ ] A new reader could follow the paper+repo instructions without hidden assumptions.

  **QA Scenarios**:
  ```
  Scenario: Repo-paper instructions match
    Tool: Read
    Preconditions: T6 edits complete
    Steps:
      1. Read the repository README reproduction sections.
      2. Read the paper setup and appendix reproducibility sections.
      3. Compare commands, paths, and expected outputs.
    Expected Result: No contradiction remains.
    Failure Indicators: Same workflow described differently across docs.
    Evidence: .sisyphus/evidence/task-t6-repro-alignment.txt

  Scenario: Reproducibility claims are bounded
    Tool: Grep + Read
    Preconditions: T6 edits complete
    Steps:
      1. Search for “reproducible”, “replicate”, “identical”, and similar terms.
      2. Inspect whether wording correctly bounds expectations to ordering/regime behavior where needed.
    Expected Result: Reproducibility language is precise and honest.
    Evidence: .sisyphus/evidence/task-t6-repro-bounds.txt
  ```

  **Commit**: NO

- [x] T7. Rewrite title, abstract, and introduction around one thesis

  **What to do**:
  - Rewrite the title if needed so it highlights the real contribution rather than sounding generic.
  - Rework abstract to foreground the problem, the exact gap, the main evidence-backed finding, and the operational takeaway.
  - Compress the introduction so the problem, gap, thesis, and contributions arrive faster and more memorably.
  - Reduce over-defensive wording and thesis diffusion.

  **Must NOT do**:
  - Do not add claims stronger than what T2 validates.
  - Do not let the intro become longer while trying to make it clearer.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: This is core academic prose rewriting.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Supports abstract/title/introduction work in existing `.tex` paper source.
    - `writing-clearly-and-concisely`: Essential for intro punch and compression.
  - **Skills Evaluated but Omitted**:
    - `dev-blog-writer`: Wrong genre; this is academic prose, not blog writing.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T8-T12)
  - **Blocks**: T13, T17
  - **Blocked By**: T1, T2

  **References**:
  - `paper/main.tex:34-54` - Title and abstract currently set the reader’s first impression.
  - `paper/sections/introduction.tex:1-30` - Current intro structure and contribution list.
  - `paper/sections/conclusion.tex:4-30` - Final landing point that intro must foreshadow.

  **Acceptance Criteria**:
  - [ ] Title, abstract, and intro all state the same central thesis.
  - [ ] Abstract includes exact gap, main finding, and operational takeaway.
  - [ ] Introduction is shorter or denser in value, not merely longer.

  **QA Scenarios**:
  ```
  Scenario: First-page story is coherent
    Tool: Read
    Preconditions: T7 edits complete
    Steps:
      1. Read title, abstract, and introduction opening through contributions.
      2. Check whether problem, gap, main result, and contributions appear in that order.
    Expected Result: A reviewer can understand the paper’s point from page 1 alone.
    Evidence: .sisyphus/evidence/task-t7-first-page-story.txt

  Scenario: Intro no longer diffuses the thesis
    Tool: Grep + Read
    Preconditions: T7 edits complete
    Steps:
      1. Search for repeated or redundant framing phrases.
      2. Inspect whether multiple competing problem statements remain.
    Expected Result: One thesis dominates the intro.
    Evidence: .sisyphus/evidence/task-t7-thesis-focus.txt
  ```

  **Commit**: NO

- [x] T8. Rewrite related work and narrative scope for sharper positioning

  **What to do**:
  - Turn related work into a support beam for the thesis rather than a parallel mini-survey.
  - Tighten `narrative_scope.tex` so it reads as the paper’s explicit claim contract.
  - Make clear why the paper is neither “just benchmark tables” nor “just a framework description”.

  **Must NOT do**:
  - Do not create a second introduction in related work.
  - Do not let narrative scope repeat results verbatim without interpretive value.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: This is manuscript framing and structure work.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`, `academic-researcher`]
    - `latex-paper-en`: Keeps edits aligned with academic paper conventions.
    - `writing-clearly-and-concisely`: Helps reduce duplication and sharpen thesis support.
    - `academic-researcher`: Helpful for honest but sharp contrast with prior work.
  - **Skills Evaluated but Omitted**:
    - `deep-research`: Not needed unless major new citations are required.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T7, T9-T12)
  - **Blocks**: T13, T17
  - **Blocked By**: T1, T5

  **References**:
  - `paper/sections/related_work.tex:6-92` - Current literature structure.
  - `paper/sections/narrative_scope.tex:1-26` - Existing claim-contract section.
  - `paper/sections/introduction.tex:12-25` - Gap/positioning that this section must reinforce.

  **Acceptance Criteria**:
  - [ ] Related work strengthens rather than dilutes the central thesis.
  - [ ] Narrative scope clearly states what is claimed, what is not claimed, and why adaptation follows.
  - [ ] The manuscript’s identity is consistent across intro, related work, and narrative scope.

  **QA Scenarios**:
  ```
  Scenario: Positioning is consistent across sections
    Tool: Read
    Preconditions: T8 edits complete
    Steps:
      1. Read the intro gap paragraph, related-work positioning paragraph, and narrative-scope opening.
      2. Compare how each describes the paper’s role.
    Expected Result: All three sections support the same identity.
    Evidence: .sisyphus/evidence/task-t8-positioning-consistency.txt

  Scenario: Narrative scope acts as claim contract
    Tool: Read
    Preconditions: T8 edits complete
    Steps:
      1. Read `paper/sections/narrative_scope.tex` in full.
      2. Verify it states bounded claims and their evidence basis.
    Expected Result: Scope section reads like an explicit claim boundary, not filler.
    Evidence: .sisyphus/evidence/task-t8-claim-contract.txt
  ```

  **Commit**: NO

- [x] T9. Rewrite experimental setup for reviewer trust and evidence clarity

  **What to do**:
  - Make setup read as trustworthy, reproducible, and appropriately bounded.
  - Emphasize what is measured at each layer, why that separation matters, and what the results do and do not support.
  - Surface evidence limitations in a way that builds trust rather than sounding apologetic.
  - Clarify whether additional repetitions/new experiments are required to support current headline claims.

  **Must NOT do**:
  - Do not hide single-host or limited-variance limitations if they remain true.
  - Do not bloat methodology with nonessential detail that belongs in appendix.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: This is methods communication and reviewer-trust work.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Supports experiment-section analysis in LaTeX papers.
    - `writing-clearly-and-concisely`: Helps communicate validity boundaries cleanly.
  - **Skills Evaluated but Omitted**:
    - `academic-researcher`: Not central unless expanding external methodological comparisons.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T7, T8, T10-T12)
  - **Blocks**: T13, T15
  - **Blocked By**: T2, T6

  **References**:
  - `paper/sections/experimental_setup.tex:6-135` - Current methodology, validity framing, and replication guidance.
  - `paper/sections/limitations.tex:4-34` - Existing limitations that should be harmonized with setup.
  - `README.md` - Repo-level reproduction commands and system context.

  **Acceptance Criteria**:
  - [ ] Setup clearly explains layer-specific constructs and evidence boundaries.
  - [ ] Limitations implied in setup match limitations stated later.
  - [ ] Reviewer can understand what would or would not falsify the main claims.

  **QA Scenarios**:
  ```
  Scenario: Methods and limitations are aligned
    Tool: Read
    Preconditions: T9 edits complete
    Steps:
      1. Read setup validity subsections and limitations section.
      2. Compare whether the same evidence bounds appear in both places.
    Expected Result: No contradiction between setup confidence and limitation statements.
    Evidence: .sisyphus/evidence/task-t9-validity-alignment.txt

  Scenario: Layer definitions are explicit
    Tool: Grep + Read
    Preconditions: T9 edits complete
    Steps:
      1. Search for Rust, Python, and vLLM measurement definitions.
      2. Confirm each layer has an explicit construct definition and use in claims.
    Expected Result: The reader can tell what each layer measures and why.
    Evidence: .sisyphus/evidence/task-t9-layer-definitions.txt
  ```

  **Commit**: NO

- [x] T10. Restructure results into a reviewer-efficient story

  **What to do**:
  - Reorder and rewrite the results section so the reader gets the main punchline fast.
  - Introduce a clear hierarchy: main result, secondary nuance, integration consequence.
  - Reduce matrix-dump feeling by making section openings and closings summarize what matters.
  - Use tables/figures to support a story, not substitute for one.

  **Must NOT do**:
  - Do not merely paraphrase every cell.
  - Do not let integration results overshadow the core finding unless they are central to the thesis.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: This is scientific storytelling and section architecture.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Supports experiment/results-section review.
    - `writing-clearly-and-concisely`: Helps create high-information section summaries.
  - **Skills Evaluated but Omitted**:
    - `chart-visualization`: Use only if new plots are needed later; first fix narrative logic.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T7-T9, T11-T12)
  - **Blocks**: T14, T17
  - **Blocked By**: T2, T4

  **References**:
  - `paper/sections/results.tex:11-236` - Entire section is the primary rewrite target.
  - `paper/sections/discussion.tex:18-49` - Policy claims that results must support.
  - `paper/sections/narrative_scope.tex:4-26` - Results should directly cash out this narrative contract.

  **Acceptance Criteria**:
  - [ ] Results section communicates the main finding before deep detail.
  - [ ] Each subsection ends with an explicit takeaway linked to the thesis.
  - [ ] Rust, Python, and vLLM layers are presented in a way that strengthens rather than scatters the story.

  **QA Scenarios**:
  ```
  Scenario: Results tell a coherent top-down story
    Tool: Read
    Preconditions: T10 edits complete
    Steps:
      1. Read subsection openings and closings in results.
      2. Check whether each subsection advances the same narrative.
    Expected Result: Reader can summarize the paper’s empirical story from subsection summaries alone.
    Evidence: .sisyphus/evidence/task-t10-results-story.txt

  Scenario: No matrix-dump behavior remains
    Tool: Read
    Preconditions: T10 edits complete
    Steps:
      1. Sample paragraphs adjacent to tables/figures.
      2. Verify that text interprets the evidence rather than restating cells.
    Expected Result: Interpretation dominates, with cell-level detail only where necessary.
    Evidence: .sisyphus/evidence/task-t10-interpretation-over-cells.txt
  ```

  **Commit**: NO

- [x] T11. Make selector/default-policy justification explicit and bounded

  **What to do**:
  - Explain clearly how the adaptive default is justified by the data.
  - State what signals it uses, what it can miss, and why it still helps operationally.
  - If needed, add an ablation/analysis plan for selector misclassifications or threshold fragility.
  - Separate “engineering policy justified by evidence” from “universally optimal selector”.

  **Must NOT do**:
  - Do not imply the selector is globally optimal if it is not.
  - Do not hide medium-scale misses or fragile rows.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: This requires careful reasoning across evidence, policy, and limitations.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Keeps the analysis tied to the manuscript’s current selector discussion.
    - `writing-clearly-and-concisely`: Helps define selector claims precisely.
  - **Skills Evaluated but Omitted**:
    - `typescript-advanced-types`: Irrelevant; this is not code typing work.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T7-T10, T12)
  - **Blocks**: T12, T13, T17
  - **Blocked By**: T2

  **References**:
  - `paper/sections/narrative_scope.tex:16-26` - Existing explanation of adaptation and `default` semantics.
  - `paper/sections/results.tex:75, 99, 129-137, 236` - Current `default` discussion across formats/layers.
  - `paper/sections/discussion.tex:18-40` - Policy claims that need to be bounded.
  - `paper/sections/limitations.tex:18-34` - Existing selector imperfection and falsification language.

  **Acceptance Criteria**:
  - [ ] The manuscript explicitly states what the selector is and is not claiming.
  - [ ] Medium-scale misses and fragility are incorporated into the selector story, not buried.
  - [ ] A reviewer can tell why adaptation is justified without inferring hidden logic.

  **QA Scenarios**:
  ```
  Scenario: Selector contract is explicit
    Tool: Read
    Preconditions: T11 edits complete
    Steps:
      1. Read selector discussion in narrative scope, results, discussion, and limitations.
      2. Verify it consistently states inputs, strengths, and misses.
    Expected Result: Selector logic is clear and bounded everywhere it appears.
    Evidence: .sisyphus/evidence/task-t11-selector-contract.txt

  Scenario: Misses are not hidden
    Tool: Grep + Read
    Preconditions: T11 edits complete
    Steps:
      1. Search for known miss cases (e.g. SmolLM3-3B, Qwen3-4B, medium SafeTensors rows).
      2. Confirm they are acknowledged in policy narrative.
    Expected Result: Known failures appear in the paper’s selector discussion.
    Evidence: .sisyphus/evidence/task-t11-miss-visibility.txt
  ```

  **Commit**: NO

- [x] T12. Tighten discussion, limitations, and conclusion

  **What to do**:
  - Make discussion read like the disciplined interpretation of results, not a second results section.
  - Ensure limitations strengthen trust while keeping the paper’s core claim intact.
  - Rewrite conclusion so it lands one memorable take-away, not five nearly equal take-aways.
  - Clarify what would change the conclusion versus what merely shifts thresholds.

  **Must NOT do**:
  - Do not let limitations swallow the contribution.
  - Do not let conclusion re-expand into a full summary of the whole paper.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: This is interpretive prose and final landing-page writing for the paper.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Supports discussion/limitations/conclusion audit.
    - `writing-clearly-and-concisely`: Helps end the paper strongly and briefly.
  - **Skills Evaluated but Omitted**:
    - `dev-blog-writer`: Wrong style for a formal paper conclusion.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T7-T11)
  - **Blocks**: T13, T17
  - **Blocked By**: T1, T2, T5, T11

  **References**:
  - `paper/sections/discussion.tex:1-49` - Current interpretation and practitioner takeaway structure.
  - `paper/sections/limitations.tex:1-34` - Current validity and boundary language.
  - `paper/sections/conclusion.tex:1-30` - Final high-level landing point of the paper.

  **Acceptance Criteria**:
  - [ ] Discussion interprets results without reopening every detail.
  - [ ] Limitations are explicit, credible, and aligned with setup.
  - [ ] Conclusion leaves one dominant operational thesis in the reader’s memory.

  **QA Scenarios**:
  ```
  Scenario: Discussion and conclusion align
    Tool: Read
    Preconditions: T12 edits complete
    Steps:
      1. Read the closing of discussion and the full conclusion.
      2. Confirm both sections land the same thesis.
    Expected Result: No new or contradictory claim appears in the conclusion.
    Evidence: .sisyphus/evidence/task-t12-landing-alignment.txt

  Scenario: Limitations support trust, not collapse
    Tool: Read
    Preconditions: T12 edits complete
    Steps:
      1. Read limitations section.
      2. Verify that bounded claims remain meaningful after limitations are stated.
    Expected Result: Limitations narrow claims without gutting the paper.
    Evidence: .sisyphus/evidence/task-t12-limitations-balance.txt
  ```

  **Commit**: NO

- [x] T13. Align cross-section terminology, claims, and section transitions

  **What to do**:
  - Standardize key terms (e.g. “adaptive default”, “explicit backend”, “policy”, “layout”, “regime”, “cold-cache”).
  - Remove cross-section drift where the same concept is named differently.
  - Improve transitions so the reader feels one continuous argument instead of stitched sections.

  **Must NOT do**:
  - Do not leave synonym drift that makes the contribution feel broader or fuzzier than it is.
  - Do not perform cosmetic edits that reintroduce claim inconsistency.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: This is a whole-manuscript coherence pass.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Keeps terminology consistent across `.tex` sections.
    - `writing-clearly-and-concisely`: Helps remove redundant transitions and wording drift.
  - **Skills Evaluated but Omitted**:
    - `academic-researcher`: Not needed for internal terminology consistency.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T14-T17)
  - **Blocks**: T15, T17
  - **Blocked By**: T1, T2, T6, T7, T8, T9, T11, T12

  **References**:
  - `paper/main.tex` - Overall section ordering and top-level framing.
  - `paper/sections/*.tex` - Cross-section terminology and transitions.

  **Acceptance Criteria**:
  - [ ] Key terms are used consistently across the paper.
  - [ ] Section endings hand off logically to the next section.
  - [ ] No section quietly changes what the paper claims to be doing.

  **QA Scenarios**:
  ```
  Scenario: Terminology is consistent
    Tool: Grep
    Preconditions: T13 edits complete
    Steps:
      1. Search for key terms and close synonyms across `paper/sections/*.tex`.
      2. Inspect whether conflicting term usage remains.
    Expected Result: Preferred terminology dominates with no confusing alternates.
    Evidence: .sisyphus/evidence/task-t13-terminology-scan.txt

  Scenario: Section handoffs are smooth
    Tool: Read
    Preconditions: T13 edits complete
    Steps:
      1. Read the last paragraph of each major section and the first paragraph of the next.
      2. Confirm a logical transition exists.
    Expected Result: The paper reads as one argument, not isolated essays.
    Evidence: .sisyphus/evidence/task-t13-transitions.txt
  ```

  **Commit**: NO

- [x] T14. Improve figures/tables/captions and final manuscript readability

  **What to do**:
  - Implement the visual/caption changes identified in T4.
  - Improve line breaks, overfull boxes, table readability, and figure captions.
  - If needed, add one summary figure/table that makes the central result easier to remember.

  **Must NOT do**:
  - Do not add ornamental visuals with no argumentative value.
  - Do not degrade arXiv compatibility with fragile figure dependencies.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: This is technical communication and readability improvement.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Supports figure/caption/format review in LaTeX.
    - `writing-clearly-and-concisely`: Helps make captions and labels information-dense.
  - **Skills Evaluated but Omitted**:
    - `frontend-design`: Not relevant to academic figure polish in LaTeX.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T13, T15-T17)
  - **Blocks**: T15, T17
  - **Blocked By**: T4, T10

  **References**:
  - `paper/sections/results.tex` - All current plots/tables and captions.
  - `paper/figures/` - External assets, if any.
  - `paper/main.log` - Existing overfull box or formatting warnings if relevant.

  **Acceptance Criteria**:
  - [ ] Captions state the takeaway clearly.
  - [ ] Any summary visual added directly supports the central thesis.
  - [ ] Readability defects materially affecting PDF quality are fixed.

  **QA Scenarios**:
  ```
  Scenario: Captions are takeaway-first
    Tool: Read
    Preconditions: T14 edits complete
    Steps:
      1. Read all result figure and table captions.
      2. Verify each caption explains why the visual matters.
    Expected Result: Captions are informative without requiring adjacent paragraph reading.
    Evidence: .sisyphus/evidence/task-t14-caption-quality.txt

  Scenario: Readability issues reduced
    Tool: Bash + Read
    Preconditions: T14 edits complete
    Steps:
      1. Compile the paper.
      2. Inspect log for major overfull/underfull or formatting warnings.
    Expected Result: No severe layout defects remain.
    Evidence: .sisyphus/evidence/task-t14-formatting-warnings.txt
  ```

  **Commit**: NO

- [x] T15. Run full LaTeX compile pass and fix format/bib/ref issues

  **What to do**:
  - Run the full build sequence.
  - Fix compile errors, citation issues, unresolved references, broken labels, and formatting regressions.
  - Ensure PDF metadata, title, abstract, and bibliography render cleanly.

  **Must NOT do**:
  - Do not accept a “mostly builds” state.
  - Do not leave reference warnings unresolved unless documented as harmless.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: This is a bounded build/debug/fix task.
  - **Skills**: [`latex-paper-en`]
    - `latex-paper-en`: Specifically covers compilation, bibliography, and formatting checks.
  - **Skills Evaluated but Omitted**:
    - `writing-clearly-and-concisely`: Secondary here; build correctness is primary.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T13, T14, T16, T17)
  - **Blocks**: T16, T17
  - **Blocked By**: T3, T9, T13, T14

  **References**:
  - `paper/README.md:7-18` - Canonical build commands.
  - `paper/main.tex` - Top-level compile source.
  - `paper/main.log`, `paper/main.blg`, `paper/main.aux`, `paper/main.out` - Generated diagnostics.

  **Acceptance Criteria**:
  - [ ] Full LaTeX build completes successfully.
  - [ ] No unresolved citations or references remain.
  - [ ] PDF is suitable for upload and review.

  **QA Scenarios**:
  ```
  Scenario: Full build passes
    Tool: Bash
    Preconditions: T15 edits complete
    Steps:
      1. Run `pdflatex -interaction=nonstopmode main.tex`.
      2. Run `bibtex main`.
      3. Run `pdflatex -interaction=nonstopmode main.tex` twice more.
      4. Check exit codes and logs.
    Expected Result: All commands succeed and produce `main.pdf`.
    Failure Indicators: Non-zero exit code, unresolved refs, broken bibliography.
    Evidence: .sisyphus/evidence/task-t15-build.log

  Scenario: PDF front matter is clean
    Tool: Read/look_at
    Preconditions: T15 build complete
    Steps:
      1. Inspect the first pages of `paper/main.pdf`.
      2. Verify title, authorship, abstract, and first references render correctly.
    Expected Result: No visible front-matter rendering defects.
    Evidence: .sisyphus/evidence/task-t15-pdf-frontmatter.txt
  ```

  **Commit**: NO

- [x] T16. Produce arXiv package and perform dry-run validation

  **What to do**:
  - Build the final arXiv upload archive from source files only.
  - Verify root contents exactly match arXiv expectations.
  - Confirm no generated junk or missing source dependency remains.
  - Perform a dry-run checklist against `paper/README.md` and the final PDF.

  **Must NOT do**:
  - Do not include generated auxiliary artifacts unless intentionally needed (e.g. `main.bbl` if chosen).
  - Do not ship a wrapper directory around `paper/`.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: This is a bounded packaging/verification task.
  - **Skills**: [`latex-paper-en`]
    - `latex-paper-en`: Covers arXiv packaging and compile expectations.
  - **Skills Evaluated but Omitted**:
    - `writing-clearly-and-concisely`: Minimal prose work here.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T13-T15, T17)
  - **Blocks**: F1-F4
  - **Blocked By**: T3, T6, T15

  **References**:
  - `paper/README.md:20-40` - arXiv packaging checklist.
  - `paper/00README.json` - top-level source declaration.
  - `paper/arxiv_abstract_plain.txt` - abstract for web form.
  - `paper/main.pdf` - final compiled output to visually inspect before upload.

  **Acceptance Criteria**:
  - [ ] Upload archive root contains only intended source files.
  - [ ] Package matches declared processor/source assumptions.
  - [ ] Final dry-run checklist is fully satisfied.

  **QA Scenarios**:
  ```
  Scenario: Archive root is correct
    Tool: Bash
    Preconditions: T16 packaging complete
    Steps:
      1. Create the archive from `paper/` source files only.
      2. List archive contents.
      3. Confirm required files are at archive root with no extra wrapper directory.
    Expected Result: Archive root matches `paper/README.md` checklist.
    Evidence: .sisyphus/evidence/task-t16-archive-contents.txt

  Scenario: Final arXiv checklist passes
    Tool: Read + Bash
    Preconditions: T16 packaging complete
    Steps:
      1. Read `paper/README.md` arXiv checklist.
      2. Confirm processor, metadata, abstract, and package structure are all satisfied.
    Expected Result: Ready for manual upload with no known blocker.
    Evidence: .sisyphus/evidence/task-t16-arxiv-checklist.txt
  ```

  **Commit**: NO

- [x] T17. Final preprint polish pass for public release quality

  **What to do**:
  - Perform one final reading pass for flow, tone, repetition, confidence, and public-facing quality.
  - Ensure the paper reads like a strong public preprint even before venue targeting.
  - Harmonize abstract/plain abstract/repo README summary language.

  **Must NOT do**:
  - Do not reopen solved methodological questions unless a serious inconsistency appears.
  - Do not introduce late large edits that threaten build/package stability.

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: This is a final prose and quality pass.
  - **Skills**: [`latex-paper-en`, `writing-clearly-and-concisely`]
    - `latex-paper-en`: Keeps the pass anchored to a LaTeX paper review workflow.
    - `writing-clearly-and-concisely`: Essential for final prose tightening.
  - **Skills Evaluated but Omitted**:
    - `copywriting`: Academic prose, not marketing language.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T13-T16)
  - **Blocks**: F1-F4
  - **Blocked By**: T3, T4, T7, T8, T10, T11, T12, T13, T14, T15

  **References**:
  - `paper/main.tex` - Top-level title/authorship/keywords framing.
  - `paper/arxiv_abstract_plain.txt` - Public-facing abstract wording.
  - `paper/README.md` - Public-facing local paper guidance.
  - `paper/sections/*.tex` - Final prose coherence across the whole manuscript.

  **Acceptance Criteria**:
  - [ ] Final paper feels cohesive, confident, and not over-defensive.
  - [ ] Public-facing summaries are consistent across PDF, arXiv text, and repo docs.
  - [ ] No obvious repetition or awkwardness remains in key sections.

  **QA Scenarios**:
  ```
  Scenario: Public-facing summaries match
    Tool: Read
    Preconditions: T17 edits complete
    Steps:
      1. Read title/abstract in `paper/main.tex`.
      2. Read `paper/arxiv_abstract_plain.txt` and the paper-related summary in `README.md`.
      3. Verify these are semantically aligned.
    Expected Result: A reader sees one coherent story across all public entry points.
    Evidence: .sisyphus/evidence/task-t17-public-summary-alignment.txt

  Scenario: Final prose polish is visible
    Tool: Read
    Preconditions: T17 edits complete
    Steps:
      1. Read abstract, intro opening, results opening, discussion opening, and conclusion.
      2. Check for repetition, hedging overload, or confidence drift.
    Expected Result: Core sections are concise, aligned, and polished.
    Evidence: .sisyphus/evidence/task-t17-final-polish.txt
  ```

  **Commit**: NO

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit okay before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Verify every must-have in this plan against the actual manuscript and arXiv package. Confirm that unsupported claims were removed or bounded, contribution identity is singular, and required evidence files exist.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Manuscript Quality and Build Review** — `unspecified-high`
  Rebuild the paper, inspect logs, check references/figures/tables, and review changed manuscript files for clarity, repetition, unsupported hype, and formatting defects.
  Output: `Build [PASS/FAIL] | Bib/Refs [PASS/FAIL] | Sections [N clean/N issues] | VERDICT`

- [x] F3. **Real QA** — `unspecified-high`
  Execute the compile command, inspect the generated PDF, create the arXiv package, and verify package root contents and absence of junk files. Save artifacts to `.sisyphus/evidence/final-qa/`.
  Output: `Compile [PASS/FAIL] | PDF checks [PASS/FAIL] | Package [PASS/FAIL] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  Ensure the manuscript improvement stayed within scope: no fabricated evidence, no venue-specific overfitting, no accidental expansion into a different paper. Confirm that claims remain tied to actual measured evidence.
  Output: `Scope [PASS/FAIL] | Claim Discipline [PASS/FAIL] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Wave 1 foundation audit**: `docs(paper): define contribution, evidence, and packaging constraints`
- **Wave 2 manuscript strengthening**: `docs(paper): sharpen narrative, results, and policy justification`
- **Wave 3 release prep**: `docs(paper): finalize arxiv manuscript and submission package`

---

## Success Criteria

### Verification Commands
```bash
cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

### Final Checklist
- [x] One unmistakable main contribution is visible in abstract, intro, results, and conclusion
- [x] Every strong claim in the paper is traceable to evidence in the measured setup
- [x] Reproducibility instructions in paper and repo no longer conflict
- [x] Final PDF is arXiv-safe and readable
- [x] Final package root matches arXiv expectations
