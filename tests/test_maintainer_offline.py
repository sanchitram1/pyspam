import numpy as np
import pandas as pd
import pytest

from feature_engineering.maintainer_offline import (
    collect_maintainers,
    handle_maintainers,
)
from feature_engineering.settings import LOW_DOWNLOAD_THRESHOLD_30D


class TestCollectMaintainers:
    """Test suite for collect_maintainers helper function."""

    def test_both_authors_and_maintainers(self):
        """Test combining authors and maintainers."""
        row = pd.Series(
            {
                "distinct_authors": ["author1@example.com", "author2@example.com"],
                "distinct_maintainers": ["maint1@example.com", "maint2@example.com"],
            }
        )
        result = collect_maintainers(row)
        # Should have 4 unique emails
        assert len(result) == 4
        assert "author1@example.com" in result
        assert "author2@example.com" in result
        assert "maint1@example.com" in result
        assert "maint2@example.com" in result

    def test_duplicate_emails(self):
        """Test that duplicates are removed when author is also maintainer."""
        row = pd.Series(
            {
                "distinct_authors": ["shared@example.com", "author1@example.com"],
                "distinct_maintainers": ["shared@example.com", "maint1@example.com"],
            }
        )
        result = collect_maintainers(row)
        # Should have 3 unique emails (shared@example.com counted once)
        assert len(result) == 3
        assert "shared@example.com" in result
        assert "author1@example.com" in result
        assert "maint1@example.com" in result

    def test_empty_authors_and_maintainers(self):
        """Test when both lists are empty."""
        row = pd.Series(
            {
                "distinct_authors": [],
                "distinct_maintainers": [],
            }
        )
        result = collect_maintainers(row)
        assert result == []

    def test_only_authors_present(self):
        """Test with only authors, no maintainers."""
        row = pd.Series(
            {
                "distinct_authors": ["author1@example.com", "author2@example.com"],
                "distinct_maintainers": [],
            }
        )
        result = collect_maintainers(row)
        assert len(result) == 2
        assert "author1@example.com" in result
        assert "author2@example.com" in result

    def test_only_maintainers_present(self):
        """Test with only maintainers, no authors."""
        row = pd.Series(
            {
                "distinct_authors": [],
                "distinct_maintainers": ["maint1@example.com", "maint2@example.com"],
            }
        )
        result = collect_maintainers(row)
        assert len(result) == 2
        assert "maint1@example.com" in result
        assert "maint2@example.com" in result

    def test_non_list_authors(self):
        """Test handling of non-list authors value."""
        row = pd.Series(
            {
                "distinct_authors": None,
                "distinct_maintainers": ["maint@example.com"],
            }
        )
        result = collect_maintainers(row)
        assert result == ["maint@example.com"]

    def test_non_list_maintainers(self):
        """Test handling of non-list maintainers value."""
        row = pd.Series(
            {
                "distinct_authors": ["author@example.com"],
                "distinct_maintainers": None,
            }
        )
        result = collect_maintainers(row)
        assert result == ["author@example.com"]

    def test_both_non_list(self):
        """Test when both are non-list values."""
        row = pd.Series(
            {
                "distinct_authors": None,
                "distinct_maintainers": None,
            }
        )
        result = collect_maintainers(row)
        assert result == []

    def test_single_author_and_maintainer(self):
        """Test with single items in each."""
        row = pd.Series(
            {
                "distinct_authors": ["author@example.com"],
                "distinct_maintainers": ["maint@example.com"],
            }
        )
        result = collect_maintainers(row)
        assert len(result) == 2

    def test_returns_list(self):
        """Test that function returns a list."""
        row = pd.Series(
            {
                "distinct_authors": ["author@example.com"],
                "distinct_maintainers": ["maint@example.com"],
            }
        )
        result = collect_maintainers(row)
        assert isinstance(result, list)

    def test_string_values_instead_of_lists(self):
        """Test handling when string values are passed instead of lists."""
        row = pd.Series(
            {
                "distinct_authors": "not_a_list",
                "distinct_maintainers": "also_not_a_list",
            }
        )
        result = collect_maintainers(row)
        # Non-list values should be treated as empty
        assert result == []


class TestHandleMaintainers:
    """Test suite for handle_maintainers main function."""

    @pytest.fixture
    def basic_df(self):
        """Create a basic test DataFrame with maintainer data."""
        return pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "distinct_authors": [
                    ["auth1@test.com"],
                    ["auth2@test.com", "auth3@test.com"],
                    ["auth1@test.com"],
                ],
                "distinct_maintainers": [
                    [],
                    ["maint1@test.com"],
                    ["maint1@test.com", "maint2@test.com"],
                ],
                "t_age_last_release_days": [10, 35, 100],
                "n_downloads_30d": [1000.0, 30.0, 200.0],
                "latest_project_urls": [[], ["url1", "url2"], ["url1"]],
            }
        )

    def test_returns_dataframe(self, basic_df):
        """Test that function returns a DataFrame."""
        result = handle_maintainers(basic_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_expected_features(self, basic_df):
        """Test that all expected features are added."""
        result = handle_maintainers(basic_df)
        expected_features = [
            "n_pkgs_by_maintainers_30d",
            "n_low_download_pkgs_by_maintainers",
            "n_latest_project_urls",
        ]
        for feature in expected_features:
            assert feature in result.columns

    def test_preserves_original_columns(self, basic_df):
        """Test that original columns are preserved."""
        original_cols = set(basic_df.columns)
        result = handle_maintainers(basic_df)
        for col in original_cols:
            assert col in result.columns

    def test_preserves_row_count(self, basic_df):
        """Test that row count is preserved."""
        result = handle_maintainers(basic_df)
        assert len(result) == len(basic_df)

    def test_features_are_numeric(self, basic_df):
        """Test that new features are numeric."""
        result = handle_maintainers(basic_df)
        assert pd.api.types.is_numeric_dtype(result["n_pkgs_by_maintainers_30d"])
        assert pd.api.types.is_numeric_dtype(result["n_low_download_pkgs_by_maintainers"])
        assert pd.api.types.is_numeric_dtype(result["n_latest_project_urls"])

    def test_features_non_negative(self, basic_df):
        """Test that feature values are non-negative."""
        result = handle_maintainers(basic_df)
        assert (result["n_pkgs_by_maintainers_30d"] >= 0).all()
        assert (result["n_low_download_pkgs_by_maintainers"] >= 0).all()
        assert (result["n_latest_project_urls"] >= 0).all()

    def test_no_nulls_in_new_features(self, basic_df):
        """Test that new features have no null values."""
        result = handle_maintainers(basic_df)
        assert not result["n_pkgs_by_maintainers_30d"].isna().any()
        assert not result["n_low_download_pkgs_by_maintainers"].isna().any()
        assert not result["n_latest_project_urls"].isna().any()

    def test_project_urls_count(self, basic_df):
        """Test that project URLs are counted correctly."""
        result = handle_maintainers(basic_df)
        assert result["n_latest_project_urls"].iloc[0] == 0
        assert result["n_latest_project_urls"].iloc[1] == 2
        assert result["n_latest_project_urls"].iloc[2] == 1

    def test_with_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(
            {
                "pkg_name": [],
                "distinct_authors": [],
                "distinct_maintainers": [],
                "t_age_last_release_days": [],
                "n_downloads_30d": [],
                "latest_project_urls": [],
            }
        )
        result = handle_maintainers(df)
        assert len(result) == 0
        assert "n_pkgs_by_maintainers_30d" in result.columns

    def test_single_row_dataframe(self):
        """Test with single row."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "distinct_authors": [["author@test.com"]],
                "distinct_maintainers": [[]],
                "t_age_last_release_days": [10],
                "n_downloads_30d": [1000.0],
                "latest_project_urls": [["url1", "url2"]],
            }
        )
        result = handle_maintainers(df)
        assert len(result) == 1
        assert result["n_latest_project_urls"].iloc[0] == 2

    def test_no_maintainers(self):
        """Test packages with no maintainers/authors."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2"],
                "distinct_authors": [[], []],
                "distinct_maintainers": [[], []],
                "t_age_last_release_days": [10, 20],
                "n_downloads_30d": [100.0, 200.0],
                "latest_project_urls": [[], []],
            }
        )
        result = handle_maintainers(df)
        # Should have 0 for all maintainer features
        assert result["n_pkgs_by_maintainers_30d"].iloc[0] == 0
        assert result["n_low_download_pkgs_by_maintainers"].iloc[0] == 0

    def test_recent_release_detection(self):
        """Test that packages with recent releases (<=30 days) are counted."""
        df = pd.DataFrame(
            {
                "pkg_name": ["recent", "old"],
                "distinct_authors": [["author@test.com"], ["author@test.com"]],
                "distinct_maintainers": [[], []],
                "t_age_last_release_days": [10, 100],  # 10 is recent, 100 is old
                "n_downloads_30d": [1000.0, 1000.0],
                "latest_project_urls": [[], []],
            }
        )
        result = handle_maintainers(df)
        # Author should have 1 recent package
        assert result["n_pkgs_by_maintainers_30d"].iloc[0] == 1

    def test_low_download_detection(self):
        """Test that low-download packages are counted correctly."""
        df = pd.DataFrame(
            {
                "pkg_name": ["low", "high"],
                "distinct_authors": [["author@test.com"], ["author@test.com"]],
                "distinct_maintainers": [[], []],
                "t_age_last_release_days": [100, 100],
                "n_downloads_30d": [
                    20.0,  # Below threshold (50)
                    100.0,  # Above threshold
                ],
                "latest_project_urls": [[], []],
            }
        )
        result = handle_maintainers(df)
        # Author should have 1 low-download package
        assert result["n_low_download_pkgs_by_maintainers"].iloc[0] == 1

    def test_null_download_counts_as_low(self):
        """Test that null download counts are treated as low (0 downloads)."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "distinct_authors": [["author@test.com"]],
                "distinct_maintainers": [[]],
                "t_age_last_release_days": [100],
                "n_downloads_30d": [np.nan],  # Null is treated as 0
                "latest_project_urls": [[]],
            }
        )
        result = handle_maintainers(df)
        # Null downloads (0) should count as low
        assert result["n_low_download_pkgs_by_maintainers"].iloc[0] == 1

    def test_shared_maintainer_multiple_packages(self):
        """Test that shared maintainer counts packages from all maintainers."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "distinct_authors": [
                    ["shared@test.com"],
                    ["shared@test.com"],
                    ["other@test.com"],
                ],
                "distinct_maintainers": [[], [], []],
                "t_age_last_release_days": [10, 15, 20],
                "n_downloads_30d": [1000.0, 1000.0, 1000.0],
                "latest_project_urls": [[], [], []],
            }
        )
        result = handle_maintainers(df)
        # First two packages have same author with recent releases
        # The maintainer should see 2 recent packages across their packages
        assert result["n_pkgs_by_maintainers_30d"].iloc[0] == 2

    def test_multiple_maintainers_same_package(self):
        """Test package with multiple maintainers."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "distinct_authors": [["author1@test.com", "author2@test.com"]],
                "distinct_maintainers": [[]],
                "t_age_last_release_days": [10],
                "n_downloads_30d": [1000.0],
                "latest_project_urls": [[]],
            }
        )
        result = handle_maintainers(df)
        # Both authors should be counted as having 1 recent package each
        # So sum across maintainers should be >= their individual counts
        assert result["n_pkgs_by_maintainers_30d"].iloc[0] >= 1

    def test_project_urls_none_value(self):
        """Test handling of None value for project URLs."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "distinct_authors": [["author@test.com"]],
                "distinct_maintainers": [[]],
                "t_age_last_release_days": [10],
                "n_downloads_30d": [1000.0],
                "latest_project_urls": [None],
            }
        )
        result = handle_maintainers(df)
        assert result["n_latest_project_urls"].iloc[0] == 0

    def test_project_urls_with_non_list(self):
        """Test handling of non-list project URLs value."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2"],
                "distinct_authors": [["author@test.com"], ["author@test.com"]],
                "distinct_maintainers": [[], []],
                "t_age_last_release_days": [10, 20],
                "n_downloads_30d": [1000.0, 2000.0],
                "latest_project_urls": ["not_a_list", []],
            }
        )
        result = handle_maintainers(df)
        assert result["n_latest_project_urls"].iloc[0] == 0
        assert result["n_latest_project_urls"].iloc[1] == 0

    def test_deterministic_output(self):
        """Test that same input produces same output."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2"],
                "distinct_authors": [["author@test.com"], ["author@test.com"]],
                "distinct_maintainers": [[], ["maint@test.com"]],
                "t_age_last_release_days": [10, 35],
                "n_downloads_30d": [1000.0, 30.0],
                "latest_project_urls": [[], ["url1"]],
            }
        )

        result1 = handle_maintainers(df.copy())
        result2 = handle_maintainers(df.copy())

        pd.testing.assert_frame_equal(
            result1[
                [
                    "n_pkgs_by_maintainers_30d",
                    "n_low_download_pkgs_by_maintainers",
                    "n_latest_project_urls",
                ]
            ],
            result2[
                [
                    "n_pkgs_by_maintainers_30d",
                    "n_low_download_pkgs_by_maintainers",
                    "n_latest_project_urls",
                ]
            ],
        )

    def test_large_dataframe(self):
        """Test with larger DataFrame."""
        n = 100
        df = pd.DataFrame(
            {
                "pkg_name": [f"pkg{i}" for i in range(n)],
                "distinct_authors": [
                    [f"author{i % 10}@test.com"] for i in range(n)
                ],
                "distinct_maintainers": [
                    [f"maint{i % 5}@test.com"] if i % 3 == 0 else [] for i in range(n)
                ],
                "t_age_last_release_days": [np.random.randint(1, 365) for _ in range(n)],
                "n_downloads_30d": [
                    np.random.uniform(0, 1000) for _ in range(n)
                ],
                "latest_project_urls": [
                    [f"url{j}" for j in range(np.random.randint(0, 5))] for _ in range(n)
                ],
            }
        )
        result = handle_maintainers(df)
        assert len(result) == n
        assert "n_pkgs_by_maintainers_30d" in result.columns
        assert (result["n_pkgs_by_maintainers_30d"] >= 0).all()

    def test_empty_maintainers_list_filtering(self):
        """Test that empty maintainer emails are filtered out."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "distinct_authors": [["author@test.com", ""]],
                "distinct_maintainers": [["", "maint@test.com"]],
                "t_age_last_release_days": [10],
                "n_downloads_30d": [1000.0],
                "latest_project_urls": [[]],
            }
        )
        result = handle_maintainers(df)
        # Should handle empty strings in maintainer lists
        assert len(result) == 1

    def test_boundary_30_day_threshold(self):
        """Test boundary condition at exactly 30 days."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2"],
                "distinct_authors": [["author@test.com"], ["author@test.com"]],
                "distinct_maintainers": [[], []],
                "t_age_last_release_days": [30, 31],  # 30 is included, 31 is not
                "n_downloads_30d": [1000.0, 1000.0],
                "latest_project_urls": [[], []],
            }
        )
        result = handle_maintainers(df)
        # At day 30, should be included (<=30)
        assert result["n_pkgs_by_maintainers_30d"].iloc[0] == 1

    def test_boundary_low_download_threshold(self):
        """Test boundary condition at LOW_DOWNLOAD_THRESHOLD."""
        threshold = LOW_DOWNLOAD_THRESHOLD_30D
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2"],
                "distinct_authors": [["author@test.com"], ["author@test.com"]],
                "distinct_maintainers": [[], []],
                "t_age_last_release_days": [100, 100],
                "n_downloads_30d": [
                    threshold - 1,  # Below threshold, should be counted
                    threshold,  # At threshold, should NOT be counted
                ],
                "latest_project_urls": [[], []],
            }
        )
        result = handle_maintainers(df)
        # Only first package should be counted as low-download
        assert result["n_low_download_pkgs_by_maintainers"].iloc[0] == 1

    def test_returns_modified_dataframe(self):
        """Test that function returns DataFrame with new features added."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "distinct_authors": [["author@test.com"]],
                "distinct_maintainers": [[]],
                "t_age_last_release_days": [10],
                "n_downloads_30d": [1000.0],
                "latest_project_urls": [[]],
            }
        )
        original_cols = df.shape[1]
        result = handle_maintainers(df)

        # Function returns a new DataFrame with merged features
        assert isinstance(result, pd.DataFrame)
        # Result should have 3 new columns added
        assert result.shape[1] == original_cols + 3
        # New columns should be present
        assert "n_pkgs_by_maintainers_30d" in result.columns
        assert "n_low_download_pkgs_by_maintainers" in result.columns
        assert "n_latest_project_urls" in result.columns

    def test_complex_maintainer_scenario(self):
        """Test complex scenario with overlapping authors/maintainers."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "distinct_authors": [
                    ["alice@test.com", "bob@test.com"],
                    ["bob@test.com", "charlie@test.com"],
                    ["alice@test.com"],
                ],
                "distinct_maintainers": [
                    ["maint1@test.com"],
                    ["maint1@test.com"],
                    [],
                ],
                "t_age_last_release_days": [10, 35, 60],
                "n_downloads_30d": [100.0, 30.0, 500.0],
                "latest_project_urls": [["url1"], ["url1", "url2"], []],
            }
        )
        result = handle_maintainers(df)

        # Check project URLs
        assert result["n_latest_project_urls"].iloc[0] == 1
        assert result["n_latest_project_urls"].iloc[1] == 2
        assert result["n_latest_project_urls"].iloc[2] == 0

        # Check features are present and valid
        assert (result["n_pkgs_by_maintainers_30d"] >= 0).all()
        assert (result["n_low_download_pkgs_by_maintainers"] >= 0).all()

    def test_all_packages_recent_release(self):
        """Test when all packages have recent releases."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "distinct_authors": [
                    ["author1@test.com"],
                    ["author1@test.com"],
                    ["author2@test.com"],
                ],
                "distinct_maintainers": [[], [], []],
                "t_age_last_release_days": [1, 2, 3],  # All recent
                "n_downloads_30d": [1000.0, 2000.0, 3000.0],
                "latest_project_urls": [[], [], []],
            }
        )
        result = handle_maintainers(df)
        # author1 should have 2 recent packages, author2 should have 1
        assert result["n_pkgs_by_maintainers_30d"].iloc[0] == 2
        assert result["n_pkgs_by_maintainers_30d"].iloc[2] == 1

    def test_all_packages_low_download(self):
        """Test when all packages have low downloads."""
        threshold = LOW_DOWNLOAD_THRESHOLD_30D
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "distinct_authors": [
                    ["author1@test.com"],
                    ["author1@test.com"],
                    ["author2@test.com"],
                ],
                "distinct_maintainers": [[], [], []],
                "t_age_last_release_days": [100, 100, 100],
                "n_downloads_30d": [
                    threshold - 10,
                    threshold - 5,
                    threshold - 1,
                ],  # All low
                "latest_project_urls": [[], [], []],
            }
        )
        result = handle_maintainers(df)
        # author1 should have 2 low-download packages
        assert result["n_low_download_pkgs_by_maintainers"].iloc[0] == 2
        # author2 should have 1
        assert result["n_low_download_pkgs_by_maintainers"].iloc[2] == 1
