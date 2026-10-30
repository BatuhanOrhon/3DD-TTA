# Search and evidence ledger — 2026-10-30

Scope: current 3DD-TTA forward citations, mechanism-level guidance/cost ideas,
actual Agent B source correspondence, and prospective experiment designs.
No model experiment or Colab operation. This is a bounded review.

## Identity and citation indexes

- Target: arXiv2411.14495v1 / WACV2025 pp1566–1576 /
  DOI10.1109/WACV61041.2025.00160; one publication.
- OpenAlex target `https://api.openalex.org/works/https://doi.org/10.48550/arXiv.2411.14495`
  succeeds and resolves W4409262973; citation count4.
- OpenAlex `https://api.openalex.org/works?filter=cites:W4409262973&per-page=100`
  succeeds: GDSD, MAMVI, SA-P2SB and robotics bin-picking. Raw JSON and SHA
  are adjacent. Results were not added to earlier counts as independent works.
- Semantic Scholar arXiv target with title/citations/externalIds fields:
  HTTP429. No pressure/repeated retry after429.
- Google Scholar cited-by cluster11097964353367270493: HTTP403. No bypass.
- Web reader could not access the index endpoints; ordinary public HTTP
  attempted once after local network authorization. Index retrieval receipt
  stores successes and failures. An earlier sandbox DNS failure was not
  interpreted as a remote-service result.

## Representative exact discovery queries

These queries were issued through web search. The ledger records query text
and primary-source decisions; it is not a complete raw search-response archive.

1. `"3DD-TTA" diffusion guidance 2026`
2. `"2411.14495" citing papers`
3. `"3DD-TTA" 2025 2026 test time adaptation uncertainty`
4. `"3DD-TTA" "2026" guidance`
5. `"3DD-TTA" "adaptive" diffusion`
6. `"Test-Time Adaptation of 3D Point Clouds via Denoising Diffusion Models" "2026"`
7. `"3DD-TTA" after:2026-10-24`
8. `"3DD-TTA" guidance loss consistency semantic`
9. `"3DD-TTA" "Dastmalchi" denoising 2026 guidance`
10. `point cloud test time adaptation diffusion guidance uncertainty 2026 loss frozen`
11. `diffusion guidance skip gradient adaptive scheduling training free guidance 2025 2026`
12. `CloudFixer input adaptation diffusion model gradient denoising objective 2024`
13. `"CloudFixer" arxiv 2024`
14. `"Adaptive Guidance: Training-free" arxiv`
15. `"diffusion" "guidance" "early stopping" 2026 uncertainty`
16. `"Test-time adaptation" "diffusion" "semantic" "frozen" 2025`
17. `"SteeringTTA" primary ICML 2025`
18. `"3DD-TTA" "guidance" "loss" -site:pith.science -site:lacuna.tiptreesystems.com -site:researchgate.net`
19. `"SATOR" "Inseok" github`
20. `"SATOR" "Sample-Adaptive Token Routing" github`
21. `"GDA" "Tsai" "Generalized Diffusion" github`
22. `"Low-Rank Test-Time Training for Pre-Trained Point Cloud Models" arxiv`
23. `"Geometry-Guided Diffusion SAR Point Cloud Denoising" July 2026`
24. `"Test-time Adaptation for Point Cloud Denoising under Unseen Noise" 2027`
25. `"MAMVI" iskong17`

Search summaries, generated review sites, awesome lists and citation-index
edges served discovery only. Method equations, status and citation inclusion
use primary paper/venue/author material. No secondary numerical speedup was
ported into an expected LION gain.

## Accepted primary evidence

- SATOR/MAMVI/TTAD/LoTT-PC/GSDTTA/Uni-Adapter: fresh actual PDF bibliography
  reference/page assertions in `citation_verification.json`. TTAD PDF[33]
  differs from HTML[7], correctly kept separate.
- Original3DD, CloudFixer and GDA actual PDFs also downloaded/extracted.
  CloudFixer p7, GDA p4, SATOR p5/p9 and MAMVI p6 were visually inspected;
  additional selected pages were rendered for reference checking.
- Adaptive Guidance: primary arXiv HTML §5/Eq7; AAAI PDF request disconnected.
  No claim that the failed PDF was inspected.
- SteeringTTA: primary arXiv HTML §3/B.5 and author publication page confirming
  PUT Workshop status; OpenReview PDF is readable by web tool but the raw
  public-HTTP download403 failed. It is not among the local PDFs.
- GDSD: indexed primary publisher text §2.3/ref37 and methods was returned
  by search; full web open429. No fresh PDF-bibliography claim. The earlier
  Crossref reference evidence exists in the25Oct audit.
- SA-P2SB/robotics: index and earlier audit only in this turn; no new method
  recommendations from inaccessible full texts.

## Author code

GitHub connector used for semantic public-source reads. Pinned excerpts are
inside the raw archive's `author_code_review.json`:

- shimazing/CloudFixer `f043e24414fcf4165fdc892abff21215c9989141`;
  adapt.py:72–180, detached denoiser target and inverse-neighborhood weights.
- iskong17/SATOR `e65983b60a83936e1e5241d339475436f1a1738d`;
  tools/tta_sator.py:106–112,170–192 and utils/misc.py:157–169;
  two predictions, H/keep-ratio routing, BN_reset.
- iskong17/MAMVI `0b553973b7123da6198b79bba4ff704b8eec914c`;
  tools/tta.py:475–637, utils/parser.py:251–288; combined loss/backward and
  confidence-LR mode. Old Inseok-kong addresses404 did not end discovery.
- TTAD official repository existence/tree read; no full implementation or
  checkpoint analysis claimed. No assets installed or model scripts executed.

## Repository evidence and preservation

Research base17dd87c on a separate worktree, production identities for11files
in `checkout_identity.json`. Main root and Agent B were read-only. Accepted
Oct25/Oct28 numerical results were read from their notes; no fresh raw-run
validation was performed. No new accuracy or runtime measurements.

Raw literature ZIP, byte size, SHA256,44members, CRC and byte-equality checks
are in `bundle_receipt.json`; it is a local ignored artifact, not Git history
or a Drive backup. Keep that ZIP if moving/removing the research worktree.
No model-test suite was needed for these documentation-only changes. Fresh
source hashes and documentation/JSON/archive checks are in `validation.json`.

No exhaustive-census, latest-publication-absence, exact reproduction,
accuracy/speed improvement or novelty claim follows from this bounded search.
