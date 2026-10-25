# Search ledger — 2026-10-25

Scope: accessible literature citing3DD-TTA through2026-10-25, emphasizing numerical comparisons on ModelNet40-C, ScanObjectNN-C and ShapeNet-C. This is a bounded search, not an exhaustive citation census. Search-engine results are discovery aids; inclusion requires actual primary reference text or a publisher-deposited exact reference. No user model/runtime experiment was performed.

## Citation-index attempts

| Route | Exact URL | Observed result | Preserved evidence |
|---|---|---|---|
| Google Scholar exact-title search | `https://scholar.google.com/scholar?q=%22Test-Time+Adaptation+of+3D+Point+Clouds+via+Denoising+Diffusion+Models%22` | Web reader inaccessible. Public HTTP200, no challenge; one citation-only title record; Turkish locale “Alıntılanma sayısı:3.” | `scholar_title.html`, retrieval timestamp/SHA/CRC |
| Google Scholar actual Cited-by link extracted from that HTML | `https://scholar.google.com/scholar?cites=11097964353367270493&as_sdt=2005&sciodt=0,5&hl=tr` | Web reader inaccessible. Public HTTP200; MAMVI, SATOR and Queen's steering-primitives thesis, three results. No CAPTCHA or bypass. | `scholar_citedby.html` |
| Semantic Scholar arXiv record | `https://api.semanticscholar.org/graph/v1/paper/ARXIV:2411.14495?fields=title,citationCount,citations.title,citations.year,citations.externalIds,citations.url` | HTTP200; reported citationCount12 and12 returned citation objects. All12 candidates checked against primary text/publisher references. | `semanticscholar_target.json` |
| Semantic Scholar venue DOI | `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/WACV61041.2025.00160?fields=title,citationCount,citations.title,citations.year,citations.externalIds,citations.url` | HTTP429; no usable second inventory. No repeated retries. | Failure in retrieval manifest |
| OpenAlex target identity | `https://api.openalex.org/works/https://doi.org/10.48550/arXiv.2411.14495` | HTTP200; resolves toW4409262973 with WACV DOI and arXiv locations merged; cited_by_count4. | `openalex_target.json` |
| OpenAlex cited-by list | `https://api.openalex.org/works?filter=cites:W4409262973&per-page=100` | HTTP200;4 results; all checked. | `openalex_citing.json` |

Do not add3+12+4. Citation indexes disagree in coverage and identities; primary LoTT-PC reference was found outside these returned inventories. Full-text citations also differ between versions: the Queen's thesis cites3DD, while companion SPeaR arXiv2609.24111 does not. A citation from one is not silently inherited by the other.

## Web query log

All queries below were issued on2026-10-25 UTC through the available web search tool. Exact strings are retained; timing within the date is not reconstructed. The full transient search responses were not all saved, so this is a query/candidate ledger, not a claim of a complete raw search-result archive. Primary documents and index HTML/JSON responses are preserved with exact retrieval timestamps.

1. `"Test-Time Adaptation of 3D Point Clouds via Denoising Diffusion Models" citations`
2. `"3DD-TTA" "results" -site:github.com`
3. `"3DD-TTA" "reproduce"`
4. `"3DD-TTA" -site:github.com -site:pith.science -site:scholarfeed.org`
5. `"3DD-TTA" "reproduced"`
6. `"3DD-TTA" "ScanObjectNN"`
7. `"3DD-TTA" "GSDTTA"`
8. `"Dastmalchi" "test" "adaptation" -site:dblp.org -site:researchgate.net -site:github.com -site:pith.science -site:scholarfeed.org -site:openaccess.thecvf.com/content/WACV2025`
9. `"2411.14495" -site:arxiv.org -site:alphaxiv.org -site:huggingface.co -site:github.com -site:deeppaper.ai`
10. `"3DD" "Purge" "Adaptation"`
11. `"3DD" "LoTT"`
12. `"Test-Time Adaptation of 3D Point Clouds via Denoising Diffusion Models" -Dastmalchi -github -pith -dblp -researchgate -scholarfeed`
13. `"Purge" "3DD-TTA"`
14. `"LoTT-PC" point clouds`
15. `"Layer Normalization" "Test-Time" point clouds`
16. `"3DD-TTA" -site:github.com "PointNeXt"`
17. `"Revisiting Layer Normalization for Point Cloud" arxiv`
18. `"Low-Rank Test-Time Training" "3DD"`
19. `"Purge-Gate" "Dastmalchi"`
20. `"Test-Time Adaptation with Steering Primitives for Realigning Representations"`
21. `"SA-P2SB" point denoising`
22. `"RGBD-to-3D Object Mesh Refinement" arxiv`
23. `"3DD-TTA" "66.1"`
24. `"SA-P2SB: a structure-aware"`
25. `"Geometry-Guided Diffusion SAR Point Cloud Denoising"`
26. `"Diffusion-Driven Point Cloud Adaption"`
27. `"3DD-TTA" "reported" -site:github.com`
28. `"GSDTTA" github`
29. `"Uni-Adapter" "Tamjidi" github`
30. `"Revisiting Layer Normalization" github`
31. `"3DD-TTA" "reproduction" -site:pith.science`

Adaptive follow-up: original/venue PDFs and supplements, arXiv HTML/PDF, CVF venues, BMVC archive, AAAI publisher, Springer preview, Crossref publisher-deposited references and authors' GitHub. No external author contact occurred.

## Candidate disposition

| Candidate family | Decision/evidence |
|---|---|
| GSDTTA /arXiv2507.18225 /ICCV2025 | One work. Full paper+supplement checked. Explicit baseline rerun statement; DGCNN3DD rows on2datasets; no ShapeNet3DD row. |
| Uni-Adapter /arXiv2511.15311 /AAAI2026 | One work. Venue paper+latest full arXiv15-page PDF checked. Explicit baseline implementation, B1/s5/Uni3D-Large;3DD step search in AppendixD/Table7. |
| MAMVI /arXiv2606.12939 /ICPR2026 Scholar entry | One work for inventory; numerical extraction is from archived arXivPDF/HTML. Reference[5] verified;3DD only related work. |
| SATOR2610.21367 | Full PDF/HTML checked. Reference[5]; no3DD numerical row in Tables1–3. |
| LoTT-PC CVPR2026 | Full primary PDF checked. Reference[5]; source context all-severity; no3DD numerical row. |
| ETTA, HATTA, TTAD, surgical registration, mesh refinement, LVLM subspaces, Queen's thesis | Full primary texts checked; exact references and pages in publication catalogue. Other tasks; no relevant3DD classification row identified. |
| SA-P2SB | Springer preview/ref43 verifies3DD citation; subscription full text unavailable. PDF endpoint returned HTML/preview, not a research PDF. Numerical content remains unassessed. |
| SAR denoising | MDPI fulltext HTTP403; publisher-deposited Crossref ref_37 verifies exact3DD DOI. No full numerical audit. |
| Robotics bin-picking | IEEE endpoint HTTP202 challenge content, not a paper; publisher-deposited Crossref ref46 verifies3DD DOI. No full numerical audit. |
| SVWA2411.01116 /WACV2025 | Context only. Predates3DD preprint; venue bibliography/full text contains no3DD citation. Own source rows and pinned source evaluator audited. |
| Purge-Gate /ICCV2025 | Context only. Full primary bibliography contains no3DD reference. Source fingerprints and MATE weight statement retained without claiming copied or independent-rerun source rows. |
| LN-TTA /WACV2026 | Context only. Full primary bibliography contains no3DD reference. Public code and source fingerprints audited. |
| BayesMM2603.22070 | Full arXiv text/PDF checked; no3DD/Dastmalchi reference. Related multimodal context, excluded from directly citing count. |
| SPeaR2609.24111 | Full PDF checked; no3DD reference, despite related thesis citation. Excluded from directly citing publication count. |
| Original3DD venue/preprint, official GitHub, author social posts | Target/reference material, not later citing works. |
| Pith/Lacuna/ScholarFeed/HyperAI/ResearchGate/dblp/awesome lists | Discovery/context only; no primary numerical reproduction evidence taken from generated summaries or leaderboards. |
| Unrelated exact-string hits, CVF programme lists, other task bibliography pages | Rejected as publications citing3DD numerically unless a primary actual reference could be verified. |

## Access and extraction failures

- Web-reader failures/403 for Scholar and several CVF pages did not imply public documents were unavailable: ordinary public HTTP returned actual HTML/PDF for Scholar/CVF. No access challenge appeared on those successful responses.
- Semantic Scholar venue query429 was recorded and not retried.
- MDPI403 and IEEE202 did not yield full texts. Crossref was used to verify published reference metadata, not to pretend full-text access.
- Springer PDF URL redirected to a subscription/cookies preview HTML. Its saved response retains the requested `.pdf` filename for raw provenance, but signature/content checks explicitly exclude it from the valid research-PDF count.
- A guessed BayesMM CVF URL returned404; the public arXiv PDF was downloaded instead.
- Search-tool GSDTTA table parsing contains wrong cells; archived PDF/render checks take precedence.
- Local Poppler was absent. Read-only extraction used an isolated temporary Python environment with PyMuPDF/pypdf; no repository dependencies were changed. Selected PDF table pages were rendered and inspected; no output PDF was authored.

## Stopping boundary

All candidates from the returned Scholar3/Semantic12/OpenAlex4 inventories were resolved to a primary reference or explicitly limited publisher reference; targeted numerical candidates were checked through accessible supplements/code. Search stopped after the above31 queries and primary follow-ups. Incomplete index coverage, paywalls, later versions and unindexed works remain limitations. A new actual primary numerical comparison with original Point-MAE assets would reopen the exact-reproduction conclusion.
