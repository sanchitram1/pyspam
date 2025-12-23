import pandas as pd
import pytest

from feature_engineering.remove_redundant import drop_redundant


class TestDropRedundant:
    """Test suite for drop_redundant function."""

    @pytest.fixture
    def df_with_all_redundant_columns(self):
        """Create DataFrame with all potentially redundant columns."""
        return pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2"],
                "latest_description": ["Desc 1", "Desc 2"],
                "latest_summary": ["Sum 1", "Sum 2"],
                "licenses": ["MIT", "Apache"],
                "latest_dependencies": [["dep1"], ["dep2"]],
                "distinct_classifiers": [["clf1"], ["clf2"]],
                "distinct_keywords": [["kw1"], ["kw2"]],
                "distinct_maintainers": [["maint@test.com"], []],
                "distinct_authors": [["author@test.com"], ["author2@test.com"]],
                "latest_project_urls": [["url1"], ["url2"]],
                "t_last_release": ["2024-01-01", "2024-01-02"],
                "t_first_release": ["2023-01-01", "2023-01-02"],
                "t_time_of_day_bucket": ["morning", "afternoon"],
                "versions": [["1.0", "2.0"], ["1.0"]],
                "kept_feature": [1, 2],
            }
        )

    def test_drops_latest_description(self):
        """Test that latest_description is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_description": ["A description"],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "latest_description" not in result.columns
        assert "kept_col" in result.columns

    def test_drops_latest_summary(self):
        """Test that latest_summary is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_summary": ["A summary"],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "latest_summary" not in result.columns

    def test_drops_licenses(self):
        """Test that licenses is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "licenses": ["MIT"],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "licenses" not in result.columns

    def test_drops_latest_dependencies(self):
        """Test that latest_dependencies is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [["dep1"]],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "latest_dependencies" not in result.columns

    def test_drops_distinct_classifiers(self):
        """Test that distinct_classifiers is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "distinct_classifiers": [["clf"]],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "distinct_classifiers" not in result.columns

    def test_drops_distinct_keywords(self):
        """Test that distinct_keywords is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "distinct_keywords": [["kw"]],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "distinct_keywords" not in result.columns

    def test_drops_distinct_maintainers(self):
        """Test that distinct_maintainers is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "distinct_maintainers": [["maint@test.com"]],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "distinct_maintainers" not in result.columns

    def test_drops_distinct_authors(self):
        """Test that distinct_authors is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "distinct_authors": [["author@test.com"]],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "distinct_authors" not in result.columns

    def test_drops_latest_project_urls(self):
        """Test that latest_project_urls is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_project_urls": [["url"]],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "latest_project_urls" not in result.columns

    def test_drops_t_last_release(self):
        """Test that t_last_release is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "t_last_release": ["2024-01-01"],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "t_last_release" not in result.columns

    def test_drops_t_first_release(self):
        """Test that t_first_release is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "t_first_release": ["2023-01-01"],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "t_first_release" not in result.columns

    def test_drops_t_time_of_day_bucket(self):
        """Test that t_time_of_day_bucket is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "t_time_of_day_bucket": ["morning"],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "t_time_of_day_bucket" not in result.columns

    def test_drops_versions(self):
        """Test that versions is dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "versions": [["1.0", "2.0"]],
                "kept_col": [1],
            }
        )
        result = drop_redundant(df)
        assert "versions" not in result.columns

    def test_preserves_non_redundant_columns(self):
        """Test that non-redundant columns are preserved."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "n_name_len": [5],
                "n_downloads_30d": [100],
                "has_repo_url": [1],
                "latest_description": ["desc"],
            }
        )
        result = drop_redundant(df)
        assert "pkg_name" in result.columns
        assert "n_name_len" in result.columns
        assert "n_downloads_30d" in result.columns
        assert "has_repo_url" in result.columns
        assert "latest_description" not in result.columns

    def test_drops_all_redundant_columns(self, df_with_all_redundant_columns):
        """Test that all redundant columns are dropped."""
        result = drop_redundant(df_with_all_redundant_columns)

        redundant_columns = [
            "latest_description",
            "latest_summary",
            "licenses",
            "latest_dependencies",
            "distinct_classifiers",
            "distinct_keywords",
            "distinct_maintainers",
            "distinct_authors",
            "latest_project_urls",
            "t_last_release",
            "t_first_release",
            "t_time_of_day_bucket",
            "versions",
        ]

        for col in redundant_columns:
            assert col not in result.columns

    def test_keeps_non_redundant_columns(self, df_with_all_redundant_columns):
        """Test that non-redundant columns are kept."""
        result = drop_redundant(df_with_all_redundant_columns)

        assert "pkg_name" in result.columns
        assert "kept_feature" in result.columns

    def test_preserves_row_count(self):
        """Test that row count is preserved."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "latest_description": ["d1", "d2", "d3"],
                "n_downloads_30d": [100, 200, 300],
            }
        )
        result = drop_redundant(df)
        assert len(result) == len(df)

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_description": ["desc"],
            }
        )
        result = drop_redundant(df)
        assert isinstance(result, pd.DataFrame)

    def test_with_missing_redundant_columns(self):
        """Test that function handles missing redundant columns gracefully."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "n_downloads_30d": [100],
                # Missing most redundant columns
            }
        )
        result = drop_redundant(df)
        assert len(result) == 1
        assert "pkg_name" in result.columns
        assert "n_downloads_30d" in result.columns

    def test_with_only_redundant_columns(self):
        """Test with only redundant columns (minus required ones)."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_description": ["desc"],
                "licenses": ["MIT"],
            }
        )
        result = drop_redundant(df)
        # pkg_name is not redundant, should be kept
        assert "pkg_name" in result.columns
        assert "latest_description" not in result.columns
        assert "licenses" not in result.columns

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(
            {
                "pkg_name": [],
                "latest_description": [],
            }
        )
        result = drop_redundant(df)
        assert len(result) == 0
        assert "pkg_name" in result.columns

    def test_single_row(self):
        """Test with single row."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_description": ["desc"],
                "n_downloads_30d": [100],
            }
        )
        result = drop_redundant(df)
        assert len(result) == 1
        assert "latest_description" not in result.columns

    def test_dataframe_not_mutated(self):
        """Test that original DataFrame is not mutated."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_description": ["desc"],
                "n_downloads_30d": [100],
            }
        )
        original_cols = set(df.columns)
        drop_redundant(df)
        # Original df should not be modified
        assert set(df.columns) == original_cols

    def test_correct_column_count(self):
        """Test that correct number of columns are dropped."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_description": ["desc"],
                "latest_summary": ["sum"],
                "licenses": ["MIT"],
                "latest_dependencies": [[]],
                "kept1": [1],
                "kept2": [2],
            }
        )
        result = drop_redundant(df)
        # Should have removed 4 columns
        assert result.shape[1] == df.shape[1] - 4

    def test_with_additional_columns(self):
        """Test with additional engineered features that should be kept."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_description": ["desc"],
                "n_lev_dist_to_top1": [5],
                "dist_embed_to_legit_desc": [0.8],
                "n_pkgs_by_maintainers_30d": [2],
                "has_dependency_to_top_brand": [1],
                "t_time_of_day": ["morning"],
                "n_latest_project_urls": [2],
            }
        )
        result = drop_redundant(df)
        # Should keep engineered features
        assert "n_lev_dist_to_top1" in result.columns
        assert "dist_embed_to_legit_desc" in result.columns
        assert "n_pkgs_by_maintainers_30d" in result.columns
        assert "has_dependency_to_top_brand" in result.columns
        assert "t_time_of_day" in result.columns
        assert "n_latest_project_urls" in result.columns
        # Should drop original
        assert "latest_description" not in result.columns

    def test_case_sensitive_column_names(self):
        """Test that column name matching is exact."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_description": ["desc"],
                "Latest_Description": ["Desc"],  # Different case
                "n_downloads_30d": [100],
            }
        )
        result = drop_redundant(df)
        # Should drop exact match only
        assert "latest_description" not in result.columns
        assert "Latest_Description" in result.columns

    def test_large_dataframe(self):
        """Test with larger DataFrame."""
        n = 1000
        df = pd.DataFrame(
            {
                "pkg_name": [f"pkg{i}" for i in range(n)],
                "latest_description": ["desc"] * n,
                "latest_summary": ["sum"] * n,
                "n_downloads_30d": list(range(n)),
            }
        )
        result = drop_redundant(df)
        assert len(result) == n
        assert "latest_description" not in result.columns
        assert "n_downloads_30d" in result.columns

    def test_deterministic_output(self):
        """Test that function is deterministic."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2"],
                "latest_description": ["desc1", "desc2"],
                "n_downloads_30d": [100, 200],
            }
        )

        result1 = drop_redundant(df.copy())
        result2 = drop_redundant(df.copy())

        pd.testing.assert_frame_equal(result1, result2)

    def test_feature_engineering_workflow(self):
        """Test typical feature engineering workflow scenario."""
        # Simulate data after all feature engineering steps
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                # Engineered features (should keep)
                "n_lev_dist_to_top1": [5],
                "n_lev_dist_to_alias": [3],
                "sim_tfidf_to_legit_centroid": [0.7],
                "dist_embed_to_legit_desc": [0.6],
                "n_pkgs_by_maintainers_30d": [2],
                "n_low_download_pkgs_by_maintainers": [1],
                "n_latest_project_urls": [2],
                "has_dependency_to_top_brand": [1],
                "has_dependency_lev_close_to_brand": [0],
                "min_dep_lev_to_brand": [10],
                "t_time_of_day": ["morning"],
                # Raw features (should keep if not redundant)
                "n_name_len": [10],
                "has_digit_in_name": [0],
                "n_downloads_30d": [100],
                # Redundant columns (should drop)
                "latest_description": ["Some description"],
                "latest_summary": ["Summary"],
                "licenses": ["MIT"],
                "latest_dependencies": [["requests"]],
                "distinct_authors": [["author@test.com"]],
                "distinct_maintainers": [["maint@test.com"]],
                "latest_project_urls": [["url1"]],
                "t_time_of_day_bucket": ["morning_06_09"],
                "versions": [["1.0", "2.0"]],
            }
        )

        result = drop_redundant(df)

        # Check engineered features are kept
        engineered_features = [
            "n_lev_dist_to_top1",
            "dist_embed_to_legit_desc",
            "n_pkgs_by_maintainers_30d",
            "has_dependency_to_top_brand",
            "t_time_of_day",
        ]
        for feat in engineered_features:
            assert feat in result.columns

        # Check redundant columns are dropped
        redundant = [
            "latest_description",
            "licenses",
            "distinct_authors",
            "t_time_of_day_bucket",
            "versions",
        ]
        for col in redundant:
            assert col not in result.columns
