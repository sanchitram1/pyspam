import numpy as np
import pandas as pd
import pytest

from feature_engineering.time_offline import bucket_to_time_of_day, handle_time


class TestBucketToTimeOfDay:
    """Test suite for bucket_to_time_of_day helper function."""

    def test_morning_bucket(self):
        """Test conversion of morning bucket."""
        result = bucket_to_time_of_day("morning_06_09")
        assert result == "morning"

    def test_afternoon_bucket(self):
        """Test conversion of afternoon bucket."""
        result = bucket_to_time_of_day("afternoon_12_17")
        assert result == "afternoon"

    def test_evening_bucket(self):
        """Test conversion of evening bucket."""
        result = bucket_to_time_of_day("evening_bucket")
        assert result == "evening"

    def test_night_bucket(self):
        """Test conversion of night bucket."""
        result = bucket_to_time_of_day("night_bucket")
        assert result == "night"

    def test_morning_uppercase(self):
        """Test case-insensitive morning detection."""
        result = bucket_to_time_of_day("MORNING_06_09")
        assert result == "morning"

    def test_afternoon_uppercase(self):
        """Test case-insensitive afternoon detection."""
        result = bucket_to_time_of_day("AFTERNOON_12_17")
        assert result == "afternoon"

    def test_evening_uppercase(self):
        """Test case-insensitive evening detection."""
        result = bucket_to_time_of_day("EVENING_BUCKET")
        assert result == "evening"

    def test_night_uppercase(self):
        """Test case-insensitive night detection."""
        result = bucket_to_time_of_day("NIGHT_BUCKET")
        assert result == "night"

    def test_mixed_case(self):
        """Test mixed case detection."""
        assert bucket_to_time_of_day("Morning_06_09") == "morning"
        assert bucket_to_time_of_day("Afternoon_12_17") == "afternoon"
        assert bucket_to_time_of_day("Evening_Bucket") == "evening"
        assert bucket_to_time_of_day("Night_Bucket") == "night"

    def test_nan_value(self):
        """Test handling of NaN value."""
        result = bucket_to_time_of_day(np.nan)
        assert result == "unknown"

    def test_none_value(self):
        """Test handling of None value."""
        result = bucket_to_time_of_day(None)
        assert result == "unknown"

    def test_empty_string(self):
        """Test handling of empty string."""
        result = bucket_to_time_of_day("")
        assert result == "unknown"

    def test_unmatched_bucket(self):
        """Test handling of unmatched bucket string."""
        result = bucket_to_time_of_day("unknown_time")
        assert result == "unknown"

    def test_numeric_nan(self):
        """Test handling of numeric NaN."""
        result = bucket_to_time_of_day(float("nan"))
        assert result == "unknown"

    def test_priority_evening_over_night(self):
        """Test that evening takes priority over night when both present."""
        # Order matters: evening is checked before night
        result = bucket_to_time_of_day("evening_night")
        assert result == "evening"

    def test_numeric_bucket_conversion(self):
        """Test that numeric values are converted to string."""
        result = bucket_to_time_of_day(123)
        assert result == "unknown"

    def test_bool_value(self):
        """Test handling of boolean value."""
        result = bucket_to_time_of_day(True)
        assert result == "unknown"

    def test_substring_matching(self):
        """Test that substring matching works."""
        assert bucket_to_time_of_day("early_morning_06_09") == "morning"
        assert bucket_to_time_of_day("late_afternoon_17_21") == "afternoon"
        assert bucket_to_time_of_day("dark_evening_18_22") == "evening"
        assert bucket_to_time_of_day("late_night_22_06") == "night"

    def test_whitespace_handling(self):
        """Test handling of whitespace in bucket strings."""
        result = bucket_to_time_of_day("  morning_06_09  ")
        # String conversion and lowercasing should handle whitespace
        assert result == "morning"

    def test_return_type_is_string(self):
        """Test that return type is always string."""
        assert isinstance(bucket_to_time_of_day("morning"), str)
        assert isinstance(bucket_to_time_of_day("evening"), str)
        assert isinstance(bucket_to_time_of_day(np.nan), str)
        assert isinstance(bucket_to_time_of_day("unknown"), str)

    def test_all_valid_outputs(self):
        """Test that only valid outputs are returned."""
        valid_outputs = {"morning", "afternoon", "evening", "night", "unknown"}
        test_inputs = [
            "morning_06_09",
            "afternoon_12_17",
            "evening_bucket",
            "night_bucket",
            np.nan,
            "random",
        ]
        for inp in test_inputs:
            result = bucket_to_time_of_day(inp)
            assert result in valid_outputs

    def test_pd_isna_detection(self):
        """Test that pd.isna correctly identifies null values."""
        # pd.isna handles various null-like values
        assert bucket_to_time_of_day(pd.NaT) == "unknown"  # pandas NaT


class TestHandleTime:
    """Test suite for handle_time main function."""

    @pytest.fixture
    def basic_df(self):
        """Create a basic test DataFrame with time buckets."""
        return pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3", "pkg4"],
                "t_time_of_day_bucket": [
                    "morning_06_09",
                    "afternoon_12_17",
                    "evening_bucket",
                    "night_22_06",
                ],
            }
        )

    def test_returns_dataframe(self, basic_df):
        """Test that function returns a DataFrame."""
        result = handle_time(basic_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_time_of_day_feature(self, basic_df):
        """Test that t_time_of_day feature is added."""
        result = handle_time(basic_df)
        assert "t_time_of_day" in result.columns

    def test_preserves_original_columns(self, basic_df):
        """Test that original columns are preserved."""
        original_cols = set(basic_df.columns)
        result = handle_time(basic_df)
        for col in original_cols:
            assert col in result.columns

    def test_preserves_row_count(self, basic_df):
        """Test that row count is preserved."""
        result = handle_time(basic_df)
        assert len(result) == len(basic_df)

    def test_correct_conversions(self, basic_df):
        """Test that conversions are correct."""
        result = handle_time(basic_df)
        assert result["t_time_of_day"].iloc[0] == "morning"
        assert result["t_time_of_day"].iloc[1] == "afternoon"
        assert result["t_time_of_day"].iloc[2] == "evening"
        assert result["t_time_of_day"].iloc[3] == "night"

    def test_with_null_values(self):
        """Test handling of null time buckets."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2"],
                "t_time_of_day_bucket": ["morning_06_09", np.nan],
            }
        )
        result = handle_time(df)
        assert result["t_time_of_day"].iloc[0] == "morning"
        assert result["t_time_of_day"].iloc[1] == "unknown"

    def test_with_unknown_buckets(self):
        """Test handling of unknown bucket values."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "t_time_of_day_bucket": ["weird_bucket"],
            }
        )
        result = handle_time(df)
        assert result["t_time_of_day"].iloc[0] == "unknown"

    def test_case_insensitive(self):
        """Test that conversion is case-insensitive."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "t_time_of_day_bucket": ["MORNING_06_09"],
            }
        )
        result = handle_time(df)
        assert result["t_time_of_day"].iloc[0] == "morning"

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(
            {
                "pkg_name": [],
                "t_time_of_day_bucket": [],
            }
        )
        result = handle_time(df)
        assert len(result) == 0
        assert "t_time_of_day" in result.columns

    def test_single_row(self):
        """Test with single row."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "t_time_of_day_bucket": ["afternoon_12_17"],
            }
        )
        result = handle_time(df)
        assert len(result) == 1
        assert result["t_time_of_day"].iloc[0] == "afternoon"

    def test_all_morning(self):
        """Test when all packages are morning releases."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "t_time_of_day_bucket": [
                    "morning_06_09",
                    "morning_06_09",
                    "morning_06_09",
                ],
            }
        )
        result = handle_time(df)
        assert (result["t_time_of_day"] == "morning").all()

    def test_all_afternoon(self):
        """Test when all packages are afternoon releases."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "t_time_of_day_bucket": [
                    "afternoon_12_17",
                    "afternoon_12_17",
                    "afternoon_12_17",
                ],
            }
        )
        result = handle_time(df)
        assert (result["t_time_of_day"] == "afternoon").all()

    def test_all_evening(self):
        """Test when all packages are evening releases."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "t_time_of_day_bucket": ["evening_bucket", "evening_bucket", "evening_bucket"],
            }
        )
        result = handle_time(df)
        assert (result["t_time_of_day"] == "evening").all()

    def test_all_night(self):
        """Test when all packages are night releases."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "t_time_of_day_bucket": ["night_22_06", "night_22_06", "night_22_06"],
            }
        )
        result = handle_time(df)
        assert (result["t_time_of_day"] == "night").all()

    def test_mixed_buckets(self):
        """Test with mixed time buckets."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3", "pkg4", "pkg5"],
                "t_time_of_day_bucket": [
                    "morning_06_09",
                    "afternoon_12_17",
                    "evening_bucket",
                    "night_22_06",
                    np.nan,
                ],
            }
        )
        result = handle_time(df)
        assert result["t_time_of_day"].iloc[0] == "morning"
        assert result["t_time_of_day"].iloc[1] == "afternoon"
        assert result["t_time_of_day"].iloc[2] == "evening"
        assert result["t_time_of_day"].iloc[3] == "night"
        assert result["t_time_of_day"].iloc[4] == "unknown"

    def test_output_dtype(self):
        """Test that t_time_of_day is string type."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "t_time_of_day_bucket": ["morning_06_09"],
            }
        )
        result = handle_time(df)
        assert result["t_time_of_day"].dtype == "object"  # string dtype

    def test_no_nulls_in_output(self):
        """Test that output contains no null values."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "t_time_of_day_bucket": ["morning_06_09", np.nan, "evening_bucket"],
            }
        )
        result = handle_time(df)
        assert not result["t_time_of_day"].isna().any()

    def test_deterministic_output(self):
        """Test that same input produces same output."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2"],
                "t_time_of_day_bucket": ["morning_06_09", "evening_bucket"],
            }
        )

        result1 = handle_time(df.copy())
        result2 = handle_time(df.copy())

        pd.testing.assert_series_equal(result1["t_time_of_day"], result2["t_time_of_day"])

    def test_large_dataframe(self):
        """Test with large DataFrame."""
        n = 1000
        buckets = [
            "morning_06_09",
            "afternoon_12_17",
            "evening_bucket",
            "night_22_06",
        ] * (n // 4)

        df = pd.DataFrame(
            {
                "pkg_name": [f"pkg{i}" for i in range(n)],
                "t_time_of_day_bucket": buckets,
            }
        )
        result = handle_time(df)
        assert len(result) == n
        assert "t_time_of_day" in result.columns
        assert not result["t_time_of_day"].isna().any()

    def test_substring_buckets(self):
        """Test with longer bucket names containing time keywords."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3", "pkg4"],
                "t_time_of_day_bucket": [
                    "early_morning_06_09",
                    "late_afternoon_17_21",
                    "dark_evening_18_22",
                    "late_night_22_06",
                ],
            }
        )
        result = handle_time(df)
        assert result["t_time_of_day"].iloc[0] == "morning"
        assert result["t_time_of_day"].iloc[1] == "afternoon"
        assert result["t_time_of_day"].iloc[2] == "evening"
        assert result["t_time_of_day"].iloc[3] == "night"

    def test_numeric_bucket_values(self):
        """Test handling of numeric bucket values."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "t_time_of_day_bucket": [123],
            }
        )
        result = handle_time(df)
        # Numeric values converted to string "123" won't match any keyword
        assert result["t_time_of_day"].iloc[0] == "unknown"

    def test_returns_modified_dataframe(self):
        """Test that function returns modified DataFrame."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "t_time_of_day_bucket": ["morning_06_09"],
            }
        )
        original_cols = df.shape[1]
        result = handle_time(df)

        # Should have 1 new column
        assert result.shape[1] == original_cols + 1
        assert isinstance(result, pd.DataFrame)
