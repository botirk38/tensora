# Learnings

- 2026-04-16: The manuscript's main evidence already supports a measurement-driven systems identity. The drift came from foregrounding Tensora as a framework contribution instead of using it as the controlled vehicle for a backend-policy claim.
- 2026-04-16: The most reusable thesis language is: backend choice depends jointly on layout and scale, so the paper should argue for a measured, format-aware adaptive policy with explicit overrides.
- 2026-04-16: SafeTensors and ServerlessLLM need to be kept in the same identity frame but not flattened into one slogan. The common story is crossover-driven policy, while the layout-specific takeaway is that sync leads only on the small SafeTensors end and never leads in the measured ServerlessLLM matrix.

- 2026-04-16 (T2): The manuscript already has appropriately bounded claim language - no overclaiming on universal terms like "always", "optimal", "general", or "universally" in abstract, intro, discussion, or conclusion that isn't properly scoped.
- 2026-04-16 (T2): Evidence mapping is already correct - all strong claims map to Tables 1-2 (Rust results), Tables 4-7 (Python/vLLM results), or are properly identified as interpretations/recommendations.
- 2026-04-16 (T2): The Limitations section (Section 5) already provides honest bounds on claims, explicitly stating scope limitations and medium-scale rows where selector trails explicit backends.
- 2026-04-16 (T2): Separation of empirical findings vs interpretations vs recommendations is already well-implemented - Results section presents raw data, Discussion interprets mechanics, Conclusion provides policy recommendations.
- 2026-04-16 (T2): Claim ledger created at .sisyphus/evidence/task-t2-claim-ledger.txt documenting all claim-to-evidence mappings.

- 2026-04-16 (T6): AUDIT COMPLETE - Paper and repository alignment check performed
- 2026-04-16 (T6): FOUND: Fix needed in appendix_reproducibility.tex - uses --fixture instead of --model-id for Rust profile binary
- 2026-04-16 (T6): FOUND: Paper incorrectly references fixtures/ directory and download_models.py that are no longer used
- 2026-04-16 (T6): VERIFIED: Reproducibility bounding language correctly distinguishes ordering/regime vs exact timing
- 2026-04-16 (T6): VERIFIED: Cold cache procedure, backend naming, vLLM version, result paths all align between paper and repo
- 2026-04-16 (T6): Evidence files created: .sisyphus/evidence/task-t6-repro-alignment.txt and task-t6-repro-bounds.txt

- 2026-04-16 (T5): Audit of related-work positioning and novelty contrast completed
- 2026-04-16 (T5): Gap statement is crisp and non-defensive - explicitly states the operational gap (which backend policy to deploy for given checkpoint under specific integration constraints)
- 2026-04-16 (T5): Primary gap statement fits in one sentence: "None of these strands, alone, supplies operators with controlled, end-to-end evidence for how to submit reads for modern LLM checkpoints on Linux when bytes must flow through Rust tensor buffers, Python bindings, and an engine such as vLLM"
- 2026-04-16 (T5): Related work has no bloat - all 7 subsections either establish what prior work does or what it DOESN'T do
- 2026-04-16 (T5): Novelty contrast is non-defensive - explicitly states "not a replacement" and "complements" rather than dismissing other work
- 2026-04-16 (T5): Evidence files created: .sisyphus/evidence/task-t5-gap-statement.txt and task-t5-related-work-focus.txt
- 2026-04-16 (T5): VERDICT - No paper modifications needed. Related work positioning is adequate.

- 2026-04-16 (T3): arXiv packaging audit complete. All required files exist (main.tex, sections/ with 12 files, bib/references.bib, 00README.json, arxiv_abstract_plain.txt). No hidden local dependencies discovered - all \input references resolved, all packages are standard TeX Live. Build instructions in README.md verified as correct.
- 2026-04-16 (T3): Abstract consistency check revealed partial mismatch. The main.tex abstract frames Tensora as "a controlled experimental vehicle" while arxiv_abstract_plain.txt frames it as "an open checkpoint-loading framework". The plain text should be updated to match the measurement-driven systems framing in main.tex. Also, main.tex has "Appendix~documented" with a broken reference.
