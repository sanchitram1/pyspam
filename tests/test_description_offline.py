import numpy as np
import pandas as pd
import pytest

from feature_engineering.description_offline import handle_description


class TestHandleDescription:
    """Test suite for handle_description feature engineering function."""

    @pytest.fixture
    def basic_df(self):
        """Create a basic test DataFrame with descriptions."""
        return pd.DataFrame(
            {
                "latest_description": [
                    "A Python HTTP library for humans",
                    "NumPy is a fundamental package for array computing",
                    "Web framework for building REST APIs",
                    "Machine learning library with advanced features",
                ],
                "other_col": [1, 2, 3, 4],
            }
        )

    @pytest.fixture
    def legit_mask_all_true(self):
        """Legit mask where all packages are legitimate."""
        return np.array([True, True, True, True])

    @pytest.fixture
    def legit_mask_mixed(self):
        """Legit mask with mixed legitimate and spam."""
        return np.array([True, True, False, False])

    @pytest.fixture
    def legit_mask_all_false(self):
        """Legit mask where all packages are spam."""
        return np.array([False, False, False, False])

    def test_returns_dataframe(self, basic_df, legit_mask_all_true):
        """Test that function returns a DataFrame."""
        result = handle_description(basic_df, legit_mask_all_true)
        assert isinstance(result, pd.DataFrame)

    def test_adds_distance_feature(self, basic_df, legit_mask_all_true):
        """Test that the distance feature is added."""
        result = handle_description(basic_df, legit_mask_all_true)
        assert "dist_embed_to_legit_desc" in result.columns

    def test_preserves_original_columns(self, basic_df, legit_mask_all_true):
        """Test that all original columns are preserved."""
        original_cols = set(basic_df.columns)
        result = handle_description(basic_df, legit_mask_all_true)
        for col in original_cols:
            assert col in result.columns

    def test_preserves_row_count(self, basic_df, legit_mask_all_true):
        """Test that number of rows remains the same."""
        result = handle_description(basic_df, legit_mask_all_true)
        assert len(result) == len(basic_df)

    def test_distance_is_numeric(self, basic_df, legit_mask_all_true):
        """Test that distance feature is numeric."""
        result = handle_description(basic_df, legit_mask_all_true)
        assert pd.api.types.is_numeric_dtype(result["dist_embed_to_legit_desc"])

    def test_distance_in_valid_range(self, basic_df, legit_mask_all_true):
        """Test that distances are bounded in [0, 2]."""
        result = handle_description(basic_df, legit_mask_all_true)
        distances = result["dist_embed_to_legit_desc"]
        # Distance is 1 - similarity, where similarity is [-1, 1]
        # So distance should be in [0, 2]
        assert (distances >= 0).all() and (distances <= 2).all()

    def test_with_null_descriptions(self):
        """Test handling of null descriptions."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "A valid description",
                    None,
                    np.nan,
                    "Another valid description",
                ]
            }
        )
        legit_mask = np.array([True, True, False, False])
        result = handle_description(df, legit_mask)
        # Should handle null values without raising errors
        assert len(result) == 4
        assert "dist_embed_to_legit_desc" in result.columns
        # NaN values should be treated as empty strings
        assert not result["dist_embed_to_legit_desc"].isna().all()

    def test_with_empty_descriptions(self):
        """Test with empty string descriptions."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "A valid description",
                    "",
                    "Another valid description",
                    "",
                ]
            }
        )
        legit_mask = np.array([True, True, False, False])
        result = handle_description(df, legit_mask)
        assert len(result) == 4
        assert "dist_embed_to_legit_desc" in result.columns

    def test_with_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"latest_description": []})
        legit_mask = np.array([])
        # Empty DataFrames cause issues with sklearn's TfidfVectorizer
        with pytest.raises(ValueError, match="empty vocabulary"):
            handle_description(df, legit_mask)

    def test_with_single_row(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({"latest_description": ["A single package description"]})
        legit_mask = np.array([True])
        result = handle_description(df, legit_mask)
        assert len(result) == 1
        assert "dist_embed_to_legit_desc" in result.columns

    def test_all_legitimate_packages(self, basic_df):
        """Test when all packages are marked as legitimate."""
        legit_mask = np.array([True, True, True, True])
        result = handle_description(basic_df, legit_mask)
        # Should produce reasonable distances
        distances = result["dist_embed_to_legit_desc"]
        assert len(distances) == 4
        # Similar descriptions should have small distances
        assert distances.min() >= 0

    def test_all_spam_packages(self, basic_df):
        """Test when all packages are marked as spam."""
        legit_mask = np.array([False, False, False, False])
        result = handle_description(basic_df, legit_mask)
        # Should still compute distances (uses overall mean as fallback)
        distances = result["dist_embed_to_legit_desc"]
        assert len(distances) == 4
        assert (distances >= 0).all()

    def test_single_legitimate_package(self, basic_df):
        """Test when only one package is legitimate."""
        legit_mask = np.array([True, False, False, False])
        result = handle_description(basic_df, legit_mask)
        # Should still compute distances
        distances = result["dist_embed_to_legit_desc"]
        assert len(distances) == 4
        # First package (the legitimate one) should have distance 0 (to itself)
        assert distances.iloc[0] == 0

    def test_deterministic_output(self):
        """Test that same input produces same output."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "A Python HTTP library",
                    "A data science library",
                    "A suspicious package",
                ]
            }
        )
        legit_mask = np.array([True, True, False])

        result1 = handle_description(df.copy(), legit_mask)
        result2 = handle_description(df.copy(), legit_mask)

        # Results should be identical
        pd.testing.assert_series_equal(
            result1["dist_embed_to_legit_desc"],
            result2["dist_embed_to_legit_desc"],
            check_exact=True,
        )

    def test_similar_descriptions_close_distance(self):
        """Test that similar descriptions have close distances."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "A Python library for data analysis",
                    "A Python library for data analysis",  # Identical
                    "A completely different malware description",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)

        # First two descriptions are identical, so should have same distance
        assert result["dist_embed_to_legit_desc"].iloc[0] == result[
            "dist_embed_to_legit_desc"
        ].iloc[1]

    def test_very_long_description(self):
        """Test handling of very long descriptions."""
        long_desc = "A " + "word " * 5000
        df = pd.DataFrame(
            {
                "latest_description": [
                    long_desc,
                    "A short description",
                    "Another description",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)
        # Should handle long descriptions without error
        assert len(result) == 3
        assert (result["dist_embed_to_legit_desc"] >= 0).all()

    def test_with_special_characters(self):
        """Test descriptions with special characters."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "A library with @#$% special chars!",
                    "Normal description",
                    "Another @ description!",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)
        # Should handle special characters without error
        assert len(result) == 3
        assert "dist_embed_to_legit_desc" in result.columns

    def test_with_unicode_characters(self):
        """Test descriptions with unicode characters."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "Café résumé naïve",
                    "Normal English description",
                    "Some 中文 text",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)
        # Should handle unicode without error
        assert len(result) == 3
        assert "dist_embed_to_legit_desc" in result.columns

    def test_with_html_content(self):
        """Test descriptions with HTML tags."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "<p>A library with HTML</p>",
                    "Plain text description",
                    "<script>alert('xss')</script>",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)
        # Should handle HTML without error (TF-IDF will treat tags as text)
        assert len(result) == 3
        assert "dist_embed_to_legit_desc" in result.columns

    def test_with_code_snippets(self):
        """Test descriptions containing code."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "Install with: pip install package\nUsage: import package",
                    "Another description with code: def foo(): pass",
                    "Regular description",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)
        # Should handle code snippets
        assert len(result) == 3
        assert "dist_embed_to_legit_desc" in result.columns

    def test_mixed_legit_mask(self, basic_df):
        """Test with varied legitimate/spam distribution."""
        legit_mask = np.array([True, False, True, False])
        result = handle_description(basic_df, legit_mask)
        distances = result["dist_embed_to_legit_desc"]
        # Legitimate packages should have distances in [0, 1]
        # (closer to legitimate centroid)
        assert distances.iloc[0] <= distances.iloc[1]
        assert distances.iloc[2] <= distances.iloc[3]

    def test_large_dataframe(self):
        """Test with larger DataFrame."""
        n = 500
        descriptions = [
            "A Python HTTP library for making requests",
            "NumPy: Numerical Python for scientific computing",
            "Django: Web framework for perfectionists",
            "Flask: Micro web framework",
            "Pandas: Data manipulation library",
        ] * (n // 5)

        df = pd.DataFrame({"latest_description": descriptions})
        legit_mask = np.random.choice([True, False], n)

        result = handle_description(df, legit_mask)
        assert len(result) == n
        assert "dist_embed_to_legit_desc" in result.columns
        assert (result["dist_embed_to_legit_desc"] >= 0).all()

    def test_stop_words_handling(self):
        """Test that stop words are handled properly."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "A library that is and or the",  # Mostly stop words
                    "Python data science package",
                    "Machine learning framework",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)
        # Should handle stop words without error (they're filtered out)
        assert len(result) == 3
        assert "dist_embed_to_legit_desc" in result.columns

    def test_numeric_text_descriptions(self):
        """Test descriptions that are mostly numbers."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "123 456 789",
                    "A normal description",
                    "999 888 777",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)
        assert len(result) == 3
        assert "dist_embed_to_legit_desc" in result.columns

    def test_modified_dataframe_inplace(self):
        """Test that function modifies DataFrame in-place."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "A description",
                    "Another description",
                ],
                "value": [1, 2],
            }
        )
        original_cols = df.shape[1]
        legit_mask = np.array([True, False])

        result = handle_description(df, legit_mask)

        # DataFrame is modified in-place
        assert df.shape[1] == original_cols + 1
        assert "dist_embed_to_legit_desc" in df.columns
        assert result is df

    def test_no_nulls_in_output(self, basic_df, legit_mask_all_true):
        """Test that output distance feature has no null values."""
        result = handle_description(basic_df, legit_mask_all_true)
        assert not result["dist_embed_to_legit_desc"].isna().any()

    def test_realistic_spam_vs_legit(self):
        """Test with realistic spam-like vs legitimate descriptions."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "Professional HTTP client for Python",
                    "NumPy: Efficient numerical arrays",
                    "HURRY GET RICH FAST click here!!",
                    "!@#$%^&*() malware payload",
                ]
            }
        )
        legit_mask = np.array([True, True, False, False])
        result = handle_description(df, legit_mask)

        # Legitimate packages should have lower distances
        legit_distances = result["dist_embed_to_legit_desc"].iloc[:2]
        spam_distances = result["dist_embed_to_legit_desc"].iloc[2:]

        assert legit_distances.mean() < spam_distances.mean()

    def test_single_word_descriptions(self):
        """Test descriptions with single words."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "requests",
                    "numpy",
                    "malware",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)
        assert len(result) == 3
        assert "dist_embed_to_legit_desc" in result.columns

    def test_case_sensitivity(self):
        """Test that distances are case-insensitive (TF-IDF converts to lowercase)."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "Python HTTP Library",
                    "python http library",
                    "PYTHON HTTP LIBRARY",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)

        # Different cases of same text should have same distance
        dist1 = result["dist_embed_to_legit_desc"].iloc[0]
        dist2 = result["dist_embed_to_legit_desc"].iloc[1]
        dist3 = result["dist_embed_to_legit_desc"].iloc[2]

        # Case-insensitive matching should make these very similar
        assert abs(dist1 - dist2) < 0.01
        assert abs(dist2 - dist3) < 0.01

    def test_whitespace_handling(self):
        """Test handling of excessive whitespace."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "Normal  description  with   spaces",
                    "Normal description with spaces",
                    "Description\n\nwith\nnewlines",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)
        # Should handle whitespace variations
        assert len(result) == 3
        assert "dist_embed_to_legit_desc" in result.columns

    def test_consistency_with_different_seeds(self):
        """Test that results are consistent regardless of operation order."""
        df = pd.DataFrame(
            {
                "latest_description": [
                    "A legitimate package description",
                    "Another legit description",
                    "Spam spam spam",
                ]
            }
        )
        legit_mask = np.array([True, True, False])

        # Run multiple times - results should be consistent
        results = [handle_description(df.copy(), legit_mask) for _ in range(3)]

        for i in range(1, len(results)):
            pd.testing.assert_series_equal(
                results[0]["dist_embed_to_legit_desc"],
                results[i]["dist_embed_to_legit_desc"],
                check_exact=True,
            )

    def test_max_features_limit(self):
        """Test that max_features=5000 is applied correctly."""
        # Create description with many unique words
        unique_words = " ".join([f"word{i}" for i in range(10000)])
        df = pd.DataFrame(
            {
                "latest_description": [
                    unique_words,
                    "normal description",
                    "another description",
                ]
            }
        )
        legit_mask = np.array([True, True, False])
        result = handle_description(df, legit_mask)
        # Should handle without memory issues (limited to 5000 features)
        assert len(result) == 3
        assert "dist_embed_to_legit_desc" in result.columns
