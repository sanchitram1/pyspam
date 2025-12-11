import numpy as np
import pandas as pd

from feature_engineering.legit_mask import create_legit_mask


class TestCreateLegitMask:
    """Test suite for create_legit_mask function."""

    def test_creates_correct_mask_with_is_spam_column(self):
        """Test that function correctly identifies legitimate packages when is_spam column exists."""
        df = pd.DataFrame(
            {
                "is_spam": [0, 1, 0, 1, 0],
                "package_name": ["pkg1", "pkg2", "pkg3", "pkg4", "pkg5"],
            }
        )
        mask = create_legit_mask(df)
        expected = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_returns_numpy_array(self):
        """Test that function returns a numpy array."""
        df = pd.DataFrame({"is_spam": [0, 1]})
        result = create_legit_mask(df)
        assert isinstance(result, np.ndarray)

    def test_returns_boolean_dtype(self):
        """Test that returned array has boolean dtype."""
        df = pd.DataFrame({"is_spam": [0, 1, 0]})
        result = create_legit_mask(df)
        assert result.dtype == bool

    def test_all_legitimate_packages(self):
        """Test when all packages are legitimate (is_spam == 0)."""
        df = pd.DataFrame({"is_spam": [0, 0, 0]})
        mask = create_legit_mask(df)
        assert all(mask)

    def test_all_spam_packages(self):
        """Test when all packages are spam (is_spam != 0)."""
        df = pd.DataFrame({"is_spam": [1, 1, 1]})
        mask = create_legit_mask(df)
        assert not any(mask)

    def test_missing_is_spam_column(self):
        """Test that all True is returned when is_spam column doesn't exist."""
        df = pd.DataFrame(
            {"package_name": ["pkg1", "pkg2", "pkg3"], "version": ["1.0", "2.0", "3.0"]}
        )
        mask = create_legit_mask(df)
        expected = np.array([True, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({"is_spam": []})
        mask = create_legit_mask(df)
        assert len(mask) == 0

    def test_empty_dataframe_no_is_spam_column(self):
        """Test with empty DataFrame without is_spam column."""
        df = pd.DataFrame()
        mask = create_legit_mask(df)
        assert len(mask) == 0

    def test_string_values_in_is_spam(self):
        """Test that string values are coerced to numeric (NaN)."""
        df = pd.DataFrame({"is_spam": ["0", "1", "invalid"]})
        mask = create_legit_mask(df)
        # "0" -> 0 (legitimate), "1" -> 1 (spam), "invalid" -> NaN (should be False == 0)
        assert mask[0] is True  # "0" == 0
        assert mask[1] is False  # "1" != 0
        # For "invalid" that becomes NaN, NaN == 0 is False
        assert mask[2] is False

    def test_float_values_in_is_spam(self):
        """Test with float values in is_spam column."""
        df = pd.DataFrame({"is_spam": [0.0, 1.5, 0.0]})
        mask = create_legit_mask(df)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(mask, expected)

    def test_nan_values_treated_as_spam(self):
        """Test that NaN values are treated as non-zero (spam)."""
        df = pd.DataFrame({"is_spam": [0, np.nan, 1]})
        mask = create_legit_mask(df)
        # NaN == 0 evaluates to False
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({"is_spam": [0]})
        mask = create_legit_mask(df)
        assert mask.shape == (1,)
        assert mask[0] is True

    def test_preserves_dataframe(self):
        """Test that function doesn't modify original DataFrame."""
        df = pd.DataFrame({"is_spam": [0, 1], "name": ["a", "b"]})
        df_copy = df.copy()
        create_legit_mask(df)
        pd.testing.assert_frame_equal(df, df_copy)

    def test_large_dataframe(self):
        """Test with larger DataFrame."""
        n = 10000
        df = pd.DataFrame({"is_spam": np.random.randint(0, 2, n)})
        mask = create_legit_mask(df)
        assert len(mask) == n
        assert mask.dtype == bool
