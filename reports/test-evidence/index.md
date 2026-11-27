# SPEC test evidence

The fast suite ran against the code published as `ca6a14c`. FT-1 and FT-2 were verified locally against the same pipeline and acceptance-test code using all three raw CSVs. Host-specific paths in the logs are abbreviated.

| Test | Result | Evidence |
|---|---|---|
| UT-29 | passed | `test_effects.py::test_fixed_effects_recovers_within_channel_effect_pooled_is_biased` · [fast log](fast.log) |
| UT-30 | passed | `test_effects.py::test_missing_targets_are_dropped_not_zeroed` · [fast log](fast.log) |
| UT-31 | passed | `test_effects.py::test_clustered_se_wider_than_naive_when_channel_effects_exist` · [fast log](fast.log) |
| UT-32 | passed | `test_effects.py::test_constant_feature_gives_nan_not_error` · [fast log](fast.log) |
| UT-33 | passed | `test_effects.py::test_usage_rates_and_design_alignment` · [fast log](fast.log) |
| UT-34 | passed | `test_effects.py::test_standardized_effect_is_coef_times_sd` · [fast log](fast.log) |
| UT-35 | passed | `test_comparison.py::test_informative_block_beats_controls_and_noise_does_not` · [fast log](fast.log) |
| UT-36 | passed | `test_comparison.py::test_test_rows_do_not_change_scaling_or_fit` · [fast log](fast.log) |
| UT-37 | passed | `test_comparison.py::test_same_rows_scored_for_every_model` · [fast log](fast.log) |
| UT-38 | passed | `test_comparison.py::test_scale_on_train_uses_train_stats` · [fast log](fast.log) |
| UT-39 | passed | `test_comparison.py::test_requires_controls_only_baseline` · [fast log](fast.log) |
| UT-40 | passed | `test_report.py::test_verdict_uses_ci_not_point_estimate` · [fast log](fast.log) |
| UT-41 | passed | `test_report.py::test_rank_and_overused` · [fast log](fast.log) |
| UT-42 | passed | `test_report.py::test_missing_slice_raises` · [fast log](fast.log) |
| UT-43 | passed | `test_report.py::test_pooled_vs_within_flags_confounded_tactic` · [fast log](fast.log) |
| UT-44 | passed | `test_report.py::test_backlash_flag` · [fast log](fast.log) |
| UT-1 | passed | `test_clean.py::test_keeps_latest_snapshot_across_countries` · [fast log](fast.log) |
| UT-2 | passed | `test_rules.py::test_clickbaity_title` · [fast log](fast.log) |
| UT-4 | passed | `test_split.py::test_group_split_has_no_shared_channels` · [fast log](fast.log) |
| UT-3 | passed | `test_rules.py::test_curly_apostrophe_and_word_boundaries` · [fast log](fast.log) |
| UT-5 | passed | `test_split.py::test_within_channel_split_tests_on_latest_videos` · [fast log](fast.log) |
| UT-6 | passed | `test_split.py::test_channel_baseline_does_not_leak_own_target` · [fast log](fast.log) |
| UT-7 | passed | `test_clean.py::test_disabled_engagement_is_nan_not_zero` · [fast log](fast.log) |
| UT-8 | not run | `test_embeddings.py::test_vectors_are_unit_length_float32` · [fast log](fast.log) |
| UT-9 | not run | `test_embeddings.py::test_caps_are_invisible` · [fast log](fast.log) |
| UT-10 | not run | `test_embeddings.py::test_different_emojis_look_alike` · [fast log](fast.log) |
| UT-11 | not run | `test_embeddings.py::test_exclamation_count_barely_matters` · [fast log](fast.log) |
| UT-12 | not run | `test_embeddings.py::test_identical_titles_get_identical_vectors` · [fast log](fast.log) |
| UT-13 | passed | `test_embeddings.py::test_cache_recomputes_when_model_changes` · [fast log](fast.log) |
| UT-14 | not run | `test_embeddings.py::test_cpu_and_accelerator_agree` · [fast log](fast.log) |
| UT-15 | passed | `test_embedding_model.py::test_ridge_ignores_test_rows` · [fast log](fast.log) |
| UT-16 | passed | `test_embedding_model.py::test_ridge_skips_missing_targets` · [fast log](fast.log) |
| UT-17 | passed | `test_embedding_model.py::test_score_known_values` · [fast log](fast.log) |
| UT-18 | passed | `test_embedding_model.py::test_clusters_fit_on_train_only` · [fast log](fast.log) |
| UT-19 | passed | `test_tfidf_model.py::test_ridge_ignores_test_rows` · [fast log](fast.log) |
| UT-20 | passed | `test_tfidf_model.py::test_ridge_skips_missing_targets` · [fast log](fast.log) |
| UT-21 | passed | `test_tfidf_model.py::test_group_cv_keeps_channels_together` · [fast log](fast.log) |
| UT-22 | passed | `test_tfidf_model.py::test_score_known_values` · [fast log](fast.log) |
| UT-23 | passed | `test_tfidf_model.py::test_controls_fit_on_train_only_and_unseen_category_ok` · [fast log](fast.log) |
| UT-24 | passed | `test_tfidf_model.py::test_scaler_ignores_test_rows` · [fast log](fast.log) |
| UT-25 | passed | `test_tfidf_model.py::test_baseline_fill_ignores_test_rows` · [fast log](fast.log) |
| UT-26 | passed | `test_tfidf_model.py::test_baseline_only_for_split_time` · [fast log](fast.log) |
| UT-27 | passed | `test_tfidf_model.py::test_top_terms_finds_planted_word` · [fast log](fast.log) |
| UT-28 | passed | `test_tfidf_model.py::test_top_terms_rejects_vocab_mismatch` · [fast log](fast.log) |
| FT-1 | passed | [Pipeline log](pipeline.log) |
| FT-2 | passed | [Pipeline log](pipeline.log) |
| FT-4 | passed, 15.18 s, 10 s budget exceeded | [Fast log](fast.log) |
| FT-5 | not run | Run with `--full` |
| FT-3 | reports available | [Held-out model metrics](../model_comparison.csv) · [OLS coefficient table](../effects.csv) |
| FT-6 | evidence pending | Three-target embedding comparison on both splits |
| FT-7 | evidence pending | Cluster report and human review |
| FT-8 | reports available | [TF-IDF metrics](../tfidf_metrics.csv) · [Top terms](../tfidf_top_terms.csv) |
