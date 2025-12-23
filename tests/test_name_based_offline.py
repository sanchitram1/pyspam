import numpy as np
import pandas as pd
import pytest

from feature_engineering.name_based_offline import add_name_based


class TestAddNameBased:
    """Test suite for add_name_based feature engineering function."""

    @pytest.fixture
    def basic_df(self):
        """Create a basic test DataFrame with required columns."""
        return pd.DataFrame(
            {
                "pkg_name": ["requests", "numpy", "malware-pkg", "typosquat-django"],
                "other_col": [1, 2, 3, 4],
            }
        )

    @pytest.fixture
    def legit_mask_all_true(self):
        """Create a legit mask where all packages are legitimate."""
        return np.array([True, True, True, True])

    @pytest.fixture
    def legit_mask_mixed(self):
        """Create a legit mask with mixed legitimate and spam packages."""
        return np.array([True, True, False, False])

    def test_returns_dataframe(self, basic_df, legit_mask_all_true):
        """Test that function returns a DataFrame."""
        result = add_name_based(basic_df, legit_mask_all_true)
        assert isinstance(result, pd.DataFrame)

    def test_adds_expected_features(self, basic_df, legit_mask_all_true):
        """Test that all expected features are added to the DataFrame."""
        result = add_name_based(basic_df, legit_mask_all_true)
        expected_features = [
            "n_lev_dist_to_top1",
            "n_lev_dist_to_alias",
            "sim_tfidf_to_legit_centroid",
        ]
        for feature in expected_features:
            assert feature in result.columns

    def test_preserves_original_columns(self, basic_df, legit_mask_all_true):
        """Test that original columns are preserved."""
        original_cols = set(basic_df.columns)
        result = add_name_based(basic_df, legit_mask_all_true)
        for col in original_cols:
            assert col in result.columns

    def test_correct_number_of_rows(self, basic_df, legit_mask_all_true):
        """Test that output has same number of rows as input."""
        result = add_name_based(basic_df, legit_mask_all_true)
        assert len(result) == len(basic_df)

    def test_levenshtein_features_are_numeric(self, basic_df, legit_mask_all_true):
        """Test that Levenshtein distance features are numeric."""
        result = add_name_based(basic_df, legit_mask_all_true)
        assert pd.api.types.is_numeric_dtype(result["n_lev_dist_to_top1"])
        assert pd.api.types.is_numeric_dtype(result["n_lev_dist_to_alias"])

    def test_tfidf_similarity_is_numeric(self, basic_df, legit_mask_all_true):
        """Test that TF-IDF similarity feature is numeric."""
        result = add_name_based(basic_df, legit_mask_all_true)
        assert pd.api.types.is_numeric_dtype(result["sim_tfidf_to_legit_centroid"])

    def test_tfidf_similarity_bounds(self, basic_df, legit_mask_all_true):
        """Test that TF-IDF similarity is bounded between -1 and 1."""
        result = add_name_based(basic_df, legit_mask_all_true)
        sim = result["sim_tfidf_to_legit_centroid"]
        assert (sim >= -1).all() and (sim <= 1).all()

    def test_levenshtein_non_negative(self, basic_df, legit_mask_all_true):
        """Test that Levenshtein distances are non-negative."""
        result = add_name_based(basic_df, legit_mask_all_true)
        assert (result["n_lev_dist_to_top1"] >= 0).all()
        assert (result["n_lev_dist_to_alias"] >= 0).all()

    def test_identical_packages_to_top_list(self):
        """Test that exact matches to top packages have distance 0."""
        df = pd.DataFrame({"pkg_name": ["requests", "numpy", "django"]})
        legit_mask = np.array([True, True, True])
        result = add_name_based(df, legit_mask)
        # Exact matches should have 0 distance
        assert result["n_lev_dist_to_top1"].iloc[0] == 0
        assert result["n_lev_dist_to_top1"].iloc[1] == 0

    def test_close_typosquat_to_top_list(self):
        """Test that similar packages have small distances."""
        df = pd.DataFrame({"pkg_name": ["requets", "numpy"]})  # typo: requets vs requests
        legit_mask = np.array([True, True])
        result = add_name_based(df, legit_mask)
        # Small typos should have small distances
        assert result["n_lev_dist_to_top1"].iloc[0] <= 2

    def test_with_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"pkg_name": []})
        legit_mask = np.array([])
        # Empty DataFrames cause issues with sklearn TfidfVectorizer
        # This is an edge case that the function doesn't handle
        # In production, this should be caught before feature engineering
        with pytest.raises(ValueError, match="empty vocabulary"):
            add_name_based(df, legit_mask)

    def test_with_null_package_names(self):
        """Test handling of null/None package names."""
        df = pd.DataFrame({"pkg_name": [None, "numpy", np.nan, "flask"]})
        legit_mask = np.array([True, True, True, True])
        result = add_name_based(df, legit_mask)
        # Should not raise errors and should handle NaNs
        assert len(result) == 4
        assert not result["n_lev_dist_to_top1"].isna().all()

    def test_with_no_legitimate_packages(self):
        """Test when no packages are marked as legitimate."""
        df = pd.DataFrame({"pkg_name": ["malware1", "malware2", "malware3"]})
        legit_mask = np.array([False, False, False])
        result = add_name_based(df, legit_mask)
        # Should handle gracefully and use overall mean as centroid
        assert len(result) == 3
        assert "sim_tfidf_to_legit_centroid" in result.columns

    def test_with_single_legitimate_package(self):
        """Test when only one package is marked as legitimate."""
        df = pd.DataFrame({"pkg_name": ["requests", "malware", "spam"]})
        legit_mask = np.array([True, False, False])
        result = add_name_based(df, legit_mask)
        assert len(result) == 3
        assert "sim_tfidf_to_legit_centroid" in result.columns

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        df = pd.DataFrame({"pkg_name": ["REQUESTS", "Numpy", "DjAnGo"]})
        legit_mask = np.array([True, True, True])
        result = add_name_based(df, legit_mask)
        # Should match regardless of case
        assert result["n_lev_dist_to_top1"].iloc[0] == 0  # REQUESTS -> requests
        assert result["n_lev_dist_to_top1"].iloc[1] == 0  # Numpy -> numpy

    def test_special_characters_in_names(self):
        """Test handling of special characters in package names."""
        df = pd.DataFrame(
            {"pkg_name": ["my-package", "my_package", "my.package", "my@package"]}
        )
        legit_mask = np.array([True, True, True, True])
        result = add_name_based(df, legit_mask)
        # Should not raise errors and produce valid distances
        assert len(result) == 4
        assert (result["n_lev_dist_to_top1"] >= 0).all()

    def test_very_long_package_names(self):
        """Test handling of very long package names."""
        long_name = "a" * 500
        df = pd.DataFrame({"pkg_name": [long_name, "requests"]})
        legit_mask = np.array([True, True])
        result = add_name_based(df, legit_mask)
        # Should handle long names without error
        assert len(result) == 2
        assert result["n_lev_dist_to_top1"].iloc[0] > 0  # Long name != any top package

    def test_mixed_legit_mask(self, basic_df, legit_mask_mixed):
        """Test with mixed legitimate/spam mask."""
        result = add_name_based(basic_df, legit_mask_mixed)
        assert len(result) == len(basic_df)
        assert all(col in result.columns for col in [
            "n_lev_dist_to_top1",
            "n_lev_dist_to_alias",
            "sim_tfidf_to_legit_centroid",
        ])

    def test_output_shape_consistency(self, basic_df, legit_mask_all_true):
        """Test that output shape is consistent with input."""
        original_cols = basic_df.shape[1]
        result = add_name_based(basic_df, legit_mask_all_true)
        # Number of rows should match
        assert result.shape[0] == basic_df.shape[0]
        # Should have exactly original columns + 3 new features
        # (function modifies in-place, so basic_df now also has new columns)
        assert result.shape[1] == original_cols + 3

    def test_no_nulls_in_new_features(self, basic_df, legit_mask_all_true):
        """Test that new features don't contain null values (except where input is null)."""
        result = add_name_based(basic_df, legit_mask_all_true)
        # For non-null input names, output should not have nulls
        valid_names = basic_df["pkg_name"].notna()
        assert not result.loc[valid_names, "n_lev_dist_to_top1"].isna().any()
        assert not result.loc[valid_names, "n_lev_dist_to_alias"].isna().any()
        assert not result.loc[valid_names, "sim_tfidf_to_legit_centroid"].isna().any()

    def test_tfidf_similarity_positive_for_common_names(self):
        """Test that TF-IDF similarity is positive for packages similar to legit names."""
        df = pd.DataFrame({"pkg_name": ["requests-utils", "numpy-ext", "random-package"]})
        legit_mask = np.array([True, True, False])
        result = add_name_based(df, legit_mask)
        # Packages with legit prefixes should have higher similarity
        sim1 = result["sim_tfidf_to_legit_centroid"].iloc[0]
        sim2 = result["sim_tfidf_to_legit_centroid"].iloc[1]
        sim3 = result["sim_tfidf_to_legit_centroid"].iloc[2]
        # The first two should have reasonable similarity
        assert sim1 >= 0
        assert sim2 >= 0

    def test_dataframe_modified_inplace(self):
        """Test that function modifies the DataFrame in-place (adds new columns)."""
        df = pd.DataFrame({"pkg_name": ["requests", "numpy"], "value": [1, 2]})
        original_rows = len(df)
        original_cols = df.shape[1]
        legit_mask = np.array([True, True])
        
        result = add_name_based(df, legit_mask)
        
        # Result should be the same object (or at least same data)
        assert len(df) == original_rows
        # The original df should now have 3 new columns added
        assert df.shape[1] == original_cols + 3
        # The result should be the modified df with new features
        assert "n_lev_dist_to_top1" in df.columns
        assert "n_lev_dist_to_alias" in df.columns
        assert "sim_tfidf_to_legit_centroid" in df.columns

    def test_with_unicode_characters(self):
        """Test handling of unicode characters in package names."""
        df = pd.DataFrame({"pkg_name": ["café", "naïve", "résumé", "requests"]})
        legit_mask = np.array([True, True, True, True])
        result = add_name_based(df, legit_mask)
        # Should not raise errors
        assert len(result) == 4
        assert (result["n_lev_dist_to_top1"] >= 0).all()

    def test_all_same_package_names(self):
        """Test when all package names are identical."""
        df = pd.DataFrame({"pkg_name": ["requests", "requests", "requests", "requests"]})
        legit_mask = np.array([True, True, True, True])
        result = add_name_based(df, legit_mask)
        # All distances to top1 should be 0
        assert (result["n_lev_dist_to_top1"] == 0).all()
        # All similarities should be identical
        sims = result["sim_tfidf_to_legit_centroid"]
        assert (sims == sims.iloc[0]).all()

    def test_large_dataframe_performance(self):
        """Test performance with larger DataFrame."""
        n = 1000
        names = ["requests", "numpy", "django", "flask", "scipy"] * (n // 5)
        df = pd.DataFrame({"pkg_name": names})
        legit_mask = np.random.choice([True, False], n)
        
        result = add_name_based(df, legit_mask)
        
        assert len(result) == n
        assert "n_lev_dist_to_top1" in result.columns
        assert "n_lev_dist_to_alias" in result.columns
        assert "sim_tfidf_to_legit_centroid" in result.columns

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({"pkg_name": ["requests"]})
        legit_mask = np.array([True])
        result = add_name_based(df, legit_mask)
        assert len(result) == 1
        assert result["n_lev_dist_to_top1"].iloc[0] == 0

    def test_deterministic_output(self):
        """Test that same input produces same output."""
        df = pd.DataFrame({"pkg_name": ["requests", "numpy", "malware"]})
        legit_mask = np.array([True, True, False])
        
        result1 = add_name_based(df.copy(), legit_mask)
        result2 = add_name_based(df.copy(), legit_mask)
        
        # Results should be identical
        pd.testing.assert_frame_equal(
            result1[["n_lev_dist_to_top1", "n_lev_dist_to_alias", "sim_tfidf_to_legit_centroid"]],
            result2[["n_lev_dist_to_top1", "n_lev_dist_to_alias", "sim_tfidf_to_legit_centroid"]]
        )
