import numpy as np
import pandas as pd
import pytest

from feature_engineering.handle_null import fill_null
from feature_engineering.settings import NUM_FILL_VALUES


class TestFillNull:
    """Test suite for fill_null function."""

    @pytest.fixture
    def basic_df(self):
        """Create a basic test DataFrame with various null types."""
        return pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "t_median_release_gap_days": [10.0, np.nan, 20.0],
                "n_downloads_7d": [100.0, 50.0, np.nan],
                "n_downloads_30d": [np.nan, 200.0, 300.0],
                "n_dependents_est": [5.0, np.nan, np.nan],
                "min_dep_lev_to_brand": [2.0, 3.0, np.nan],
                "cat_license_family": ["MIT", np.nan, "Apache"],
            }
        )

    def test_returns_dataframe(self, basic_df):
        """Test that function returns a DataFrame."""
        result = fill_null(basic_df)
        assert isinstance(result, pd.DataFrame)

    def test_fills_numeric_nulls(self):
        """Test that numeric nulls are filled with configured values."""
        df = pd.DataFrame(
            {
                "t_median_release_gap_days": [10.0, np.nan],
                "n_downloads_7d": [100.0, np.nan],
                "n_downloads_30d": [np.nan, 200.0],
                "n_dependents_est": [5.0, np.nan],
                "min_dep_lev_to_brand": [2.0, np.nan],
            }
        )
        result = fill_null(df)

        # Check that NaNs are filled with configured values
        assert not result["t_median_release_gap_days"].isna().any()
        assert not result["n_downloads_7d"].isna().any()
        assert not result["n_downloads_30d"].isna().any()
        assert not result["n_dependents_est"].isna().any()
        assert not result["min_dep_lev_to_brand"].isna().any()

    def test_correct_fill_values_for_numeric_columns(self):
        """Test that correct fill values are used for numeric columns."""
        df = pd.DataFrame(
            {
                "t_median_release_gap_days": [np.nan],
                "n_downloads_7d": [np.nan],
                "n_downloads_30d": [np.nan],
                "n_dependents_est": [np.nan],
                "min_dep_lev_to_brand": [np.nan],
            }
        )
        result = fill_null(df)

        # Verify against NUM_FILL_VALUES
        for col, fill_value in NUM_FILL_VALUES.items():
            if col in result.columns:
                assert result[col].iloc[0] == fill_value

    def test_t_median_release_gap_days_fill_value(self):
        """Test that t_median_release_gap_days is filled with 0."""
        df = pd.DataFrame({"t_median_release_gap_days": [np.nan, np.nan]})
        result = fill_null(df)
        assert (result["t_median_release_gap_days"] == 0).all()

    def test_n_downloads_7d_fill_value(self):
        """Test that n_downloads_7d is filled with 0."""
        df = pd.DataFrame({"n_downloads_7d": [np.nan, np.nan]})
        result = fill_null(df)
        assert (result["n_downloads_7d"] == 0).all()

    def test_n_downloads_30d_fill_value(self):
        """Test that n_downloads_30d is filled with 0."""
        df = pd.DataFrame({"n_downloads_30d": [np.nan, np.nan]})
        result = fill_null(df)
        assert (result["n_downloads_30d"] == 0).all()

    def test_n_dependents_est_fill_value(self):
        """Test that n_dependents_est is filled with 0."""
        df = pd.DataFrame({"n_dependents_est": [np.nan, np.nan]})
        result = fill_null(df)
        assert (result["n_dependents_est"] == 0).all()

    def test_min_dep_lev_to_brand_fill_value(self):
        """Test that min_dep_lev_to_brand is filled with 20."""
        df = pd.DataFrame({"min_dep_lev_to_brand": [np.nan, np.nan]})
        result = fill_null(df)
        assert (result["min_dep_lev_to_brand"] == 20).all()

    def test_fills_categorical_license_family_with_unknown(self):
        """Test that cat_license_family nulls are filled with 'unknown'."""
        df = pd.DataFrame(
            {
                "cat_license_family": ["MIT", np.nan, "Apache", None],
            }
        )
        result = fill_null(df)
        assert not result["cat_license_family"].isna().any()
        assert result["cat_license_family"].iloc[1] == "unknown"
        assert result["cat_license_family"].iloc[3] == "unknown"

    def test_replaces_other_or_unknown_with_unknown(self):
        """Test that 'other_or_unknown' is replaced with 'unknown'."""
        df = pd.DataFrame(
            {
                "cat_license_family": ["MIT", "other_or_unknown", "Apache"],
            }
        )
        result = fill_null(df)
        assert "other_or_unknown" not in result["cat_license_family"].values
        assert result["cat_license_family"].iloc[1] == "unknown"

    def test_preserves_valid_license_values(self):
        """Test that valid license values are preserved."""
        df = pd.DataFrame(
            {
                "cat_license_family": ["MIT", "Apache", "GPL", "BSD"],
            }
        )
        result = fill_null(df)
        assert "MIT" in result["cat_license_family"].values
        assert "Apache" in result["cat_license_family"].values
        assert "GPL" in result["cat_license_family"].values
        assert "BSD" in result["cat_license_family"].values

    def test_handles_mixed_null_types(self, basic_df):
        """Test handling of mixed null types (NaN, None)."""
        df = pd.DataFrame(
            {
                "n_downloads_30d": [100.0, np.nan, None],
                "cat_license_family": ["MIT", None, np.nan],
            }
        )
        result = fill_null(df)
        assert not result["n_downloads_30d"].isna().any()
        assert not result["cat_license_family"].isna().any()

    def test_preserves_valid_numeric_values(self):
        """Test that valid numeric values are preserved."""
        df = pd.DataFrame(
            {
                "n_downloads_30d": [100.0, 200.5, 300],
                "t_median_release_gap_days": [10.0, 20.5, 30],
            }
        )
        result = fill_null(df)
        assert result["n_downloads_30d"].iloc[0] == 100.0
        assert result["n_downloads_30d"].iloc[1] == 200.5
        assert result["t_median_release_gap_days"].iloc[2] == 30

    def test_preserves_row_count(self, basic_df):
        """Test that row count is preserved."""
        result = fill_null(basic_df)
        assert len(result) == len(basic_df)

    def test_with_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(
            {
                "n_downloads_30d": [],
                "cat_license_family": [],
            }
        )
        result = fill_null(df)
        assert len(result) == 0

    def test_with_no_nulls(self):
        """Test with DataFrame containing no nulls."""
        df = pd.DataFrame(
            {
                "n_downloads_30d": [100.0, 200.0, 300.0],
                "cat_license_family": ["MIT", "Apache", "GPL"],
            }
        )
        result = fill_null(df)
        # Should remain unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_with_all_nulls_in_column(self):
        """Test with column containing all nulls."""
        df = pd.DataFrame(
            {
                "n_downloads_30d": [np.nan, np.nan, np.nan],
            }
        )
        result = fill_null(df)
        assert (result["n_downloads_30d"] == 0).all()

    def test_with_missing_numeric_columns(self):
        """Test when some numeric columns are missing."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "n_downloads_30d": [np.nan],
                # Other columns from NUM_FILL_VALUES are missing
            }
        )
        result = fill_null(df)
        # Should handle gracefully
        assert result["n_downloads_30d"].iloc[0] == 0

    def test_with_missing_license_column(self):
        """Test when license column is missing."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "n_downloads_30d": [np.nan],
                # cat_license_family is missing
            }
        )
        result = fill_null(df)
        # Should handle gracefully
        assert len(result) == 1

    def test_with_additional_columns(self):
        """Test that additional columns are preserved."""
        df = pd.DataFrame(
            {
                "n_downloads_30d": [np.nan],
                "cat_license_family": [np.nan],
                "extra_column": ["value"],
                "another_column": [123],
            }
        )
        result = fill_null(df)
        assert "extra_column" in result.columns
        assert "another_column" in result.columns
        assert result["extra_column"].iloc[0] == "value"
        assert result["another_column"].iloc[0] == 123

    def test_string_numbers_converted_to_numeric(self):
        """Test that string numbers are converted to numeric."""
        df = pd.DataFrame(
            {
                "n_downloads_30d": ["100", "200", "invalid"],
            }
        )
        result = fill_null(df)
        # "invalid" becomes NaN, then filled with 0
        assert result["n_downloads_30d"].iloc[0] == 100.0
        assert result["n_downloads_30d"].iloc[1] == 200.0
        assert result["n_downloads_30d"].iloc[2] == 0.0  # "invalid" -> NaN -> 0

    def test_returns_dataframe_copy(self):
        """Test that function returns result (may or may not be same object)."""
        df = pd.DataFrame(
            {
                "n_downloads_30d": [np.nan],
            }
        )
        result = fill_null(df)
        assert isinstance(result, pd.DataFrame)

    def test_deterministic_output(self):
        """Test that same input produces same output."""
        df = pd.DataFrame(
            {
                "n_downloads_30d": [100.0, np.nan, 300.0],
                "cat_license_family": ["MIT", None, "Apache"],
            }
        )

        result1 = fill_null(df.copy())
        result2 = fill_null(df.copy())

        pd.testing.assert_frame_equal(result1, result2)

    def test_numeric_conversion_with_errors(self):
        """Test numeric conversion with coercion of errors."""
        df = pd.DataFrame(
            {
                "t_median_release_gap_days": ["10", "20", "invalid", None],
            }
        )
        result = fill_null(df)
        # "invalid" should become NaN then filled with 0
        assert result["t_median_release_gap_days"].iloc[0] == 10.0
        assert result["t_median_release_gap_days"].iloc[1] == 20.0
        assert result["t_median_release_gap_days"].iloc[2] == 0.0
        assert result["t_median_release_gap_days"].iloc[3] == 0.0

    def test_license_standardization(self):
        """Test that license family standardization works correctly."""
        df = pd.DataFrame(
            {
                "cat_license_family": [
                    "MIT",
                    "other_or_unknown",
                    "Apache",
                    "other_or_unknown",
                    None,
                    np.nan,
                ],
            }
        )
        result = fill_null(df)
        expected = pd.Series(
            ["MIT", "unknown", "Apache", "unknown", "unknown", "unknown"],
            name="cat_license_family",
        )
        pd.testing.assert_series_equal(result["cat_license_family"], expected)

    def test_large_dataframe(self):
        """Test with large DataFrame."""
        n = 1000
        df = pd.DataFrame(
            {
                "n_downloads_30d": [100.0 if i % 2 == 0 else np.nan for i in range(n)],
                "cat_license_family": [
                    "MIT"
                    if i % 3 == 0
                    else ("other_or_unknown" if i % 3 == 1 else None)
                    for i in range(n)
                ],
            }
        )
        result = fill_null(df)
        assert len(result) == n
        assert not result["n_downloads_30d"].isna().any()
        assert not result["cat_license_family"].isna().any()

    def test_edge_case_zero_values(self):
        """Test that zero values are preserved (not treated as null)."""
        df = pd.DataFrame(
            {
                "n_downloads_30d": [0.0, 100.0, np.nan],
                "t_median_release_gap_days": [0, 10, None],
            }
        )
        result = fill_null(df)
        assert result["n_downloads_30d"].iloc[0] == 0.0  # Preserved
        assert result["n_downloads_30d"].iloc[1] == 100.0
        assert result["n_downloads_30d"].iloc[2] == 0.0  # Filled
        assert result["t_median_release_gap_days"].iloc[0] == 0  # Preserved

    def test_float_and_int_mixing(self):
        """Test handling of mixed float and int values."""
        df = pd.DataFrame(
            {
                "n_downloads_30d": [100, 200.5, np.nan, 400],
            }
        )
        result = fill_null(df)
        assert result["n_downloads_30d"].iloc[0] == 100
        assert result["n_downloads_30d"].iloc[1] == 200.5
        assert result["n_downloads_30d"].iloc[2] == 0.0
        assert result["n_downloads_30d"].iloc[3] == 400

    def test_negative_numeric_values(self):
        """Test that negative numeric values are preserved."""
        df = pd.DataFrame(
            {
                "min_dep_lev_to_brand": [-5, -10, np.nan],
            }
        )
        result = fill_null(df)
        # Note: negative values may not make sense for this column,
        # but function should preserve them
        assert result["min_dep_lev_to_brand"].iloc[0] == -5
        assert result["min_dep_lev_to_brand"].iloc[1] == -10
        assert result["min_dep_lev_to_brand"].iloc[2] == 20  # Filled

    def test_full_pipeline_simulation(self):
        """Test a full pipeline simulation with mixed data."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "t_median_release_gap_days": [10.0, np.nan, 30.0],
                "n_downloads_7d": [100.0, np.nan, 300.0],
                "n_downloads_30d": [np.nan, 200.0, 300.0],
                "n_dependents_est": [5.0, 10.0, np.nan],
                "min_dep_lev_to_brand": [2.0, np.nan, 5.0],
                "cat_license_family": ["MIT", "other_or_unknown", np.nan],
                "other_feature": ["a", "b", "c"],
            }
        )

        result = fill_null(df)

        # Check no nulls in numeric columns
        numeric_cols = [
            "t_median_release_gap_days",
            "n_downloads_7d",
            "n_downloads_30d",
            "n_dependents_est",
            "min_dep_lev_to_brand",
        ]
        for col in numeric_cols:
            assert not result[col].isna().any()

        # Check no nulls in license column
        assert not result["cat_license_family"].isna().any()

        # Check other feature preserved
        assert result["other_feature"].tolist() == ["a", "b", "c"]

        # Verify specific values
        assert result["n_downloads_7d"].iloc[1] == 0  # Was NaN
        assert (
            result["cat_license_family"].iloc[1] == "unknown"
        )  # Was "other_or_unknown"
        assert result["cat_license_family"].iloc[2] == "unknown"  # Was NaN
