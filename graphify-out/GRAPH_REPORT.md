# Graph Report - knowledgeMap  (2026-11-18)

## Corpus Check
- 122 files · ~331,983 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 7 file(s) not represented in the graph (top: (none) 2, .jsonl 2, .ini 1)

## Summary
- 1995 nodes · 6169 edges · 88 communities (65 shown, 23 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 628 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ffb1f366`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- validation.py
- cytoscape.min.js
- create_app
- KnowledgeService
- KGStore
- typing
- test_parsers.py
- app.js
- textutil.py
- enrichment.py
- segmentation.py
- KnowledgeApplication
- interfaces.py
- test_browser_governed.py
- observability.py
- engine.py
- .ingest_extracted_directory
- pdf_extraction.py
- merging.py
- test_kg_extraction.py
- .project
- project_filter
- GovernedGraph
- Pipeline
- test_lifecycle.py
- services.py
- test_api.py
- test_metadata_images.py
- store_relationships
- router.py
- test_kg_pipeline.py
- test_governance.py
- text_formats.py
- claims.py
- test_projects.py
- test_contracts_runtime.py
- NotFound
- job
- lifespan
- ProjectStore
- llm.py
- test_knowledge_engine.py
- mcp_server.py
- ModelRouter
- units.py
- MetadataStore
- locks.py
- export_vault
- canonical.py
- test_hybrid_retrieval.py
- Any
- retriever
- app.py
- envcheck.py
- Any
- ResourceGate
- test_upserts_never_set_a_field_twice_which_real_mongodb_rejects
- benchmark.py
- FastEmbedder
- test_knowledge_lifecycle.py
- stable_id
- _UnionFind
- ingest_pdf
- evaluation.py
- extract_images
- StageTracker
- types.js
- json
- .handle
- MongoKnowledgeStore
- test_architecture.py
- WatchEvent
- test_abbreviations.py
- SourceRef
- GlinerExtractor
- VitCaptioner

## God Nodes (most connected - your core abstractions)
1. `create_app()` - 103 edges
2. `project_filter()` - 87 edges
3. `KnowledgeApplication` - 83 edges
4. `MongoKnowledgeStore` - 82 edges
5. `main()` - 54 edges
6. `services()` - 51 edges
7. `MongoKnowledgeRepository` - 49 edges
8. `MetadataStore` - 47 edges
9. `call()` - 47 edges
10. `stable_id()` - 42 edges

## Surprising Connections (you probably didn't know these)
- `bad_request()` --calls--> `LLMError`  [INFERRED]
  tests/test_kg_extraction.py → src/contracts/interfaces.py
- `fatal()` --calls--> `LLMFatalError`  [INFERRED]
  tests/test_kg_extraction.py → src/contracts/interfaces.py
- `test_concept_detail_by_canonical_id_shows_names_documents_and_merge_history()` --uses--> `NotFound`  [INFERRED]
  tests/test_governed_concepts.py → src/orchestration/services.py
- `test_agent_proposals_enter_the_graph_only_through_governance()` --uses--> `NotFound`  [INFERRED]
  tests/test_knowledge_engine.py → src/orchestration/services.py
- `test_reviewer_split_and_merge_are_reversible_and_recorded()` --uses--> `NotFound`  [INFERRED]
  tests/test_knowledge_engine.py → src/orchestration/services.py

## Import Cycles
- None detected.

## Communities (88 total, 23 thin omitted)

### Community 0 - "validation.py"
Cohesion: 0.07
Nodes (33): best(), chunk_offset_to_page(), domain_expansion(), Evidence, LocalConcept, LocalRelationship, worst(), fence() (+25 more)

### Community 1 - "cytoscape.min.js"
Cohesion: 0.05
Nodes (58): a(), Ao(), b(), Ba(), cs(), d(), dc(), ds() (+50 more)

### Community 2 - "create_app"
Cohesion: 0.10
Nodes (42): create_app(), adjudicate(), agent_activity(), call(), claims(), conflicts(), create_project(), diagnostics() (+34 more)

### Community 3 - "KnowledgeService"
Cohesion: 0.07
Nodes (25): is_noise(), chunk_text_by_words(), CitationError, KnowledgeService, phrase_match(), insert_pages(), page_id(), _extract() (+17 more)

### Community 5 - "typing"
Cohesion: 0.08
Nodes (5): note_name(), Captioner, link_images_to_concepts(), upsert_all(), is_tombstoned()

### Community 6 - "test_parsers.py"
Cohesion: 0.08
Nodes (15): DocumentParser, ContentUnit, ParsedDocument, UnitType, _bbox(), DoclingParser, PyMuPDFStructureParser, add() (+7 more)

### Community 7 - "app.js"
Cohesion: 0.13
Nodes (38): api(), citationButton(), currentProject, documentsCache, ensureGraph(), escapeHtml(), fillDocumentFilter(), GLOBAL_ENDPOINTS (+30 more)

### Community 8 - "textutil.py"
Cohesion: 0.11
Nodes (14): clean_surface(), collapse_whitespace(), _is_boundary(), is_toc_page(), _limit_span(), mask_dot_leaders(), normalize_concept(), split_sentences() (+6 more)

### Community 9 - "enrichment.py"
Cohesion: 0.06
Nodes (37): EnrichmentConfig, alias_candidates(), build_concepts(), build_relationships(), _choose_description(), cluster(), kept_apart(), collect_groups() (+29 more)

### Community 10 - "segmentation.py"
Cohesion: 0.13
Nodes (23): ChunkingConfig, Block, _blocks_from_spans(), _build_chunk(), cohesion(), is_heading_line(), _normalize_line(), page_blocks() (+15 more)

### Community 11 - "KnowledgeApplication"
Cohesion: 0.13
Nodes (3): main(), _page(), KnowledgeApplication

### Community 12 - "interfaces.py"
Cohesion: 0.08
Nodes (24): KnowledgeRepository, KnowledgeSet, UnitRecord, Adjudication, CheckResult, Claim, Concept, Document (+16 more)

### Community 13 - "test_browser_governed.py"
Cohesion: 0.13
Nodes (7): ScriptedText, _free_port(), Page, server(), _session(), test_clicking_a_governed_node_shows_its_canonical_record_and_never_a_stale_one(), scenario()

### Community 14 - "observability.py"
Cohesion: 0.25
Nodes (3): configure(), JsonFormatter, test_json_logs_carry_context_fields()

### Community 15 - "engine.py"
Cohesion: 0.17
Nodes (8): analyze(), build_graph(), graph_stats(), load_graph(), main(), weighted_pagerank(), word_spans(), test_graph_keeps_evidence_and_computes_analysis()

### Community 17 - "pdf_extraction.py"
Cohesion: 0.10
Nodes (10): _extract_pdf(), _has_usable_text(), main(), ocr_image_file(), _ocr_page(), _page_text(), _resolve_tesseract(), run() (+2 more)

### Community 18 - "merging.py"
Cohesion: 0.08
Nodes (22): Embedder, looks_like_abbreviation(), block_reason(), build_nodes(), _Cached, Candidate, _choose_description(), cluster_candidates() (+14 more)

### Community 19 - "test_kg_extraction.py"
Cohesion: 0.09
Nodes (17): extract_chunk(), extraction_is_current(), repair_prompt(), user_prompt(), parse_output(), RecoveryResult, run_with_recovery(), structure_errors() (+9 more)

### Community 20 - ".project"
Cohesion: 0.13
Nodes (5): _names(), test_agent_proposals_enter_the_graph_only_through_governance(), test_ontology_extension_through_services_is_recorded(), test_reviewer_split_and_merge_are_reversible_and_recorded(), test_application_services_adjudication_isolation_and_recovery()

### Community 21 - "project_filter"
Cohesion: 0.08
Nodes (5): HistoryEntry, project_filter(), MongoKnowledgeRepository, _sequence(), test_structure_units_are_added_without_touching_pages_or_other_projects()

### Community 22 - "GovernedGraph"
Cohesion: 0.09
Nodes (6): HybridRetriever, evidence_payload(), GovernedGraph, group_claims(), passages(), SemanticIndex

### Community 23 - "Pipeline"
Cohesion: 0.13
Nodes (4): Pipeline, test_app_lifecycle_delete_also_updates_the_concept_first_graph(), test_deleting_a_document_only_changes_its_project(), two_projects()

### Community 24 - "test_lifecycle.py"
Cohesion: 0.16
Nodes (12): sha256_file(), FolderWatcher, process_events(), _ids_everywhere(), make_pdf(), test_add_then_delete_removes_every_derived_record(), test_deleted_document_is_not_re_added_by_the_watcher(), test_document_with_another_source_survives_watch_removal() (+4 more)

### Community 25 - "services.py"
Cohesion: 0.08
Nodes (12): Ontology, OntologyType, Contract, ExtractionResult, KnowledgeItem, Proposal, utcnow(), _answer() (+4 more)

### Community 26 - "test_api.py"
Cohesion: 0.07
Nodes (9): client(), FakePipeline, test_delete_and_replace_endpoints(), test_page_text_and_rendered_image(), test_upload_runs_a_background_job(), _wait(), _ApiPipeline, test_api_projects_and_isolation() (+1 more)

### Community 27 - "test_metadata_images.py"
Cohesion: 0.14
Nodes (10): _body_size(), detect_metadata(), _largest_text(), _lines(), _title_case_if_shouting(), _make_book(), _make_pdf_with_image(), test_detects_title_editor_and_year_with_sources() (+2 more)

### Community 28 - "store_relationships"
Cohesion: 0.14
Nodes (11): build_relationships(), pattern_for(), detect_relation(), _name_pattern(), npmi(), store_relationships(), _id(), test_detect_relation_patterns_and_direction() (+3 more)

### Community 29 - "router.py"
Cohesion: 0.11
Nodes (6): SpanModel, TextModel, VisionModel, GatedSpans, GatedText, GatedVision

### Community 30 - "test_kg_pipeline.py"
Cohesion: 0.15
Nodes (15): ScriptedLLM, Judge, test_decisions_belong_to_one_project(), _answer(), corpus(), _pipeline(), _snapshot(), test_changed_page_text_re_extracts_only_its_chunk() (+7 more)

### Community 31 - "test_governance.py"
Cohesion: 0.32
Nodes (15): KnowledgeStatus, claim(), concept(), context(), ev(), failed(), run(), test_adjudication_overrides_the_automatic_status() (+7 more)

### Community 32 - "text_formats.py"
Cohesion: 0.07
Nodes (24): assess(), _has_text(), _collapse(), _HTML, read(), read_docx(), read_html(), read_markdown() (+16 more)

### Community 33 - "claims.py"
Cohesion: 0.18
Nodes (12): modality_hint(), uncertainty_markers(), weaker(), build_claims(), claim_structure_errors(), claim_user_prompt(), claims_are_current(), extract_claims_chunk() (+4 more)

### Community 34 - "test_projects.py"
Cohesion: 0.18
Nodes (6): _counts(), pipeline(), projects(), test_deleting_a_project_removes_only_that_project(), test_identical_concepts_in_two_projects_never_merge_or_connect(), test_migration_only_adds_the_default_project()

### Community 35 - "test_contracts_runtime.py"
Cohesion: 0.15
Nodes (8): RuntimeConfig, LLMError, LLMFatalError, httpx_transport(), ollama_status(), OllamaModel, _text_of(), test_ollama_text_and_vision_follow_the_shared_protocols()

### Community 36 - "NotFound"
Cohesion: 0.13
Nodes (3): NotFound, test_graph_exploration_paths_and_governed_visibility(), test_adjudication_can_be_revoked_and_both_decisions_stay_in_history()

### Community 37 - "job"
Cohesion: 0.20
Nodes (15): concept_graph(), delete_document(), delete_project_endpoint(), job(), run(), rebuild_project(), replace_file(), save_upload() (+7 more)

### Community 38 - "lifespan"
Cohesion: 0.15
Nodes (5): lifespan(), page_image(), _highlight_rects(), JobManager, WatchService

### Community 39 - "ProjectStore"
Cohesion: 0.19
Nodes (5): is_default(), _clean_name(), ProjectError, ProjectStore, test_project_management()

### Community 40 - "llm.py"
Cohesion: 0.23
Nodes (3): LLMConfig, AnthropicClient, create_client()

### Community 41 - "test_knowledge_engine.py"
Cohesion: 0.13
Nodes (19): KGConfig, KGPipeline, FakeEmbedder, insert_corpus(), test_one_canonical_concept_per_term_across_documents(), _run(), test_abbreviation_across_documents_becomes_one_node_with_provenance(), test_ambiguous_abbreviation_is_left_for_review_not_guessed() (+11 more)

### Community 42 - "mcp_server.py"
Cohesion: 0.13
Nodes (9): main(), MCPServer, _tool_result(), AgentGateway, AgentPolicy, Grant, _grant(), PermissionDenied (+1 more)

### Community 43 - "ModelRouter"
Cohesion: 0.26
Nodes (5): ModelRouter, ModelUnavailable, FakeTransport, test_local_models_run_one_at_a_time_and_unload_on_switch(), test_router_routes_capabilities_without_model_specific_callers()

### Community 44 - "units.py"
Cohesion: 0.24
Nodes (10): Quantity, contains_quantity(), convert(), dimension(), find_quantities(), normalize_unit(), requested_unit(), same_value() (+2 more)

### Community 45 - "MetadataStore"
Cohesion: 0.24
Nodes (4): _clean(), MetadataStore, metadata(), test_user_edits_survive_new_detection()

### Community 46 - "locks.py"
Cohesion: 0.18
Nodes (4): ProjectBusy, ProjectLocks, test_one_writer_per_project_across_threads_and_expired_locks(), _try()

### Community 47 - "export_vault"
Cohesion: 0.20
Nodes (5): export_vault(), _yaml(), format_citation(), test_obsidian_vault_notes(), test_citation_format_uses_metadata_and_unknown()

### Community 48 - "canonical.py"
Cohesion: 0.28
Nodes (14): Evidence, Provenance, Relationship, content_unit(), _evidence(), gliner_concept(), gliner_relationship(), kg_claim() (+6 more)

### Community 49 - "test_hybrid_retrieval.py"
Cohesion: 0.09
Nodes (18): fuse(), app(), _concept(), KeywordEmbedder, test_evidence_is_rechecked_against_the_stored_source(), test_hybrid_search_combines_methods_and_every_quote_is_an_exact_slice(), test_reciprocal_rank_fusion_rewards_agreement(), test_research_separates_facts_interpretations_conflicts_and_converts_units() (+10 more)

### Community 51 - "retriever"
Cohesion: 0.22
Nodes (9): ask(), autocomplete(), concept(), concept_search(), health(), page(), retriever(), search() (+1 more)

### Community 52 - "app.py"
Cohesion: 0.19
Nodes (9): AdjudicationBody, AskRequest, _default_app(), MergeBody, MetadataUpdate, OntologyBody, ProjectBody, ProposalBody (+1 more)

### Community 53 - "envcheck.py"
Cohesion: 0.20
Nodes (8): in_virtualenv(), main(), mismatches(), pinned_versions(), report(), warn_if_needed(), test_reads_pins_including_extras(), test_reports_wrong_and_missing_packages()

### Community 57 - "benchmark.py"
Cohesion: 0.23
Nodes (5): benchmark_text_model(), _percentile(), process_rss_mb(), sample(), run_benchmark()

### Community 59 - "test_knowledge_lifecycle.py"
Cohesion: 0.17
Nodes (11): LLMResponse, FakeText, test_benchmark_measures_latency_failures_memory_and_quality(), complete(), corpus(), FakeTextModel, FakeVision, _kg() (+3 more)

### Community 61 - "stable_id"
Cohesion: 0.18
Nodes (12): scoped_id(), scoped_key(), stable_id(), merge_concepts(), write_report(), reviewer_record_id(), _mention(), _names_by_node() (+4 more)

### Community 63 - "ingest_pdf"
Cohesion: 0.15
Nodes (9): document_id_for(), delete_document(), delete_project(), ingest_pdf(), ingest_text_file(), _inside(), is_ingested(), remove_source() (+1 more)

### Community 64 - "evaluation.py"
Cohesion: 0.29
Nodes (6): evaluate_dataset(), evaluate_extraction(), evaluate_knowledge_citations(), evaluate_retrieval(), _page_lookup(), test_evaluation_metrics_on_a_labelled_corpus()

### Community 65 - "extract_images"
Cohesion: 0.33
Nodes (6): _blocks(), extract_images(), find_caption_and_context(), _horizontal_overlap(), _save_image(), _vertical_gap()

### Community 68 - "types.js"
Cohesion: 0.32
Nodes (11): badge(), contrast(), cssVars(), derive(), dot(), hexToRgb(), legend(), luminance() (+3 more)

### Community 69 - "json"
Cohesion: 0.14
Nodes (9): _print(), backup(), restore(), _sha256(), rpc(), test_api_answers_409_while_another_process_writes(), test_mcp_tools_route_through_services_with_citations(), test_project_backup_restores_into_an_empty_database_and_never_overwrites() (+1 more)

### Community 70 - ".handle"
Cohesion: 0.31
Nodes (3): _error(), _schema(), test_mcp_protocol_handshake_and_tool_listing()

### Community 71 - "MongoKnowledgeStore"
Cohesion: 0.10
Nodes (20): _embedder(), main(), _alpha_words(), build_page_mentions(), configure_logging(), ensure_indexes(), extract_concepts(), main() (+12 more)

### Community 74 - "test_architecture.py"
Cohesion: 0.39
Nodes (4): import_edges(), test_components_only_import_what_their_boundary_allows(), test_contracts_are_independent_of_models_parsers_and_storage(), test_the_legacy_exception_list_has_no_stale_entries()

### Community 76 - "test_abbreviations.py"
Cohesion: 0.18
Nodes (8): concept(), relation(), _chunk(), _one(), test_context_dependent_and_guessed_expansions_go_to_review(), test_standard_expansions_are_accepted_from_the_abbreviation_alone(), test_unresolved_abbreviations_stay_as_they_are(), test_validation_statuses()

### Community 78 - "SourceRef"
Cohesion: 0.33
Nodes (3): SourceRef, _evidence(), test_knowledge_items_keep_project_isolation_and_source_offsets()

### Community 79 - "GlinerExtractor"
Cohesion: 0.13
Nodes (6): ChunkModel, GlinerExtractor, FakeGliner, test_extractor_maps_chunk_offsets_back_to_the_page(), test_replace_records_a_version_lineage(), pipeline()

## Knowledge Gaps
- **23 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `create_app()` connect `create_app` to `KnowledgeService`, `KnowledgeApplication`, `test_browser_governed.py`, `test_lifecycle.py`, `test_api.py`, `text_formats.py`, `test_projects.py`, `NotFound`, `job`, `lifespan`, `ProjectStore`, `test_knowledge_engine.py`, `MetadataStore`, `test_hybrid_retrieval.py`, `retriever`, `app.py`, `envcheck.py`, `test_knowledge_lifecycle.py`, `json`, `MongoKnowledgeStore`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `create_app()` (e.g. with `lifespan()` and `ProjectError`) actually correct?**
  _`create_app()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Should `validation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07291666666666667 - nodes in this community are weakly interconnected._
- **Why does `KnowledgeApplication` connect `KnowledgeApplication` to `create_app`, `StageTracker`, `NotFound`, `typing`, `lifespan`, `ProjectStore`, `test_knowledge_engine.py`, `mcp_server.py`, `GlinerExtractor`, `test_hybrid_retrieval.py`, `Any`, `app.py`, `project_filter`, `.project`, `services.py`, `test_knowledge_lifecycle.py`, `ingest_pdf`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `KnowledgeApplication` (e.g. with `MCPServer` and `_page()`) actually correct?**
  _`KnowledgeApplication` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `cytoscape.min.js` be split into smaller, more focused modules?**
  _Cohesion score 0.05067920585161965 - nodes in this community are weakly interconnected._
- **Why does `MongoKnowledgeStore` connect `MongoKnowledgeStore` to `create_app`, `KnowledgeService`, `KGStore`, `typing`, `KnowledgeApplication`, `test_browser_governed.py`, `engine.py`, `.ingest_extracted_directory`, `merging.py`, `project_filter`, `Pipeline`, `test_lifecycle.py`, `services.py`, `store_relationships`, `router.py`, `lifespan`, `test_knowledge_engine.py`, `mcp_server.py`, `MetadataStore`, `export_vault`, `app.py`, `.__init__`, `stable_id`, `ingest_pdf`, `extract_images`, `StageTracker`, `json`, `test_abbreviations.py`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._