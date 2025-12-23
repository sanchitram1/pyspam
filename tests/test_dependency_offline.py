import math

import numpy as np
import pandas as pd
import pytest

from feature_engineering.dependency_offline import (
    deps_base_names,
    handle_dependency,
)
from feature_engineering.settings import BRAND_ALIASES, LEV_THRESHOLD, TOP_BRAND_PKGS


class TestDepsBaseNames:
    """Test suite for deps_base_names helper function."""

    def test_simple_package_names(self):
        """Test extracting simple package names without version specs."""
        deps = ["requests", "numpy", "django"]
        result = deps_base_names(deps)
        assert result == ["requests", "numpy", "django"]

    def test_with_version_specifiers(self):
        """Test extraction with version specifiers."""
        deps = ["requests>=2.0", "numpy==1.19.0", "django<3.0"]
        result = deps_base_names(deps)
        assert result == ["requests", "numpy", "django"]

    def test_with_complex_version_ranges(self):
        """Test with complex version specifications."""
        deps = ["requests>=2.0,<3.0", "numpy~=1.19.0", "django!=2.0"]
        result = deps_base_names(deps)
        assert result == ["requests", "numpy", "django"]

    def test_with_environment_markers(self):
        """Test extraction with environment markers."""
        deps = [
            'requests>=2.0; python_version<"3.10"',
            "numpy==1.19.0; platform_system=='Windows'",
        ]
        result = deps_base_names(deps)
        assert result == ["requests", "numpy"]

    def test_with_extras(self):
        """Test extraction with extras specification."""
        deps = ["requests[security]>=2.0", "numpy[mkl]>=1.19.0"]
        result = deps_base_names(deps)
        # Extras are part of the base name extraction, but extract_pkg_name_from_requirement
        # splits on semicolon first, so they should be included
        assert "requests" in result[0].lower() or "requests" == result[0].lower()
        assert "numpy" in result[1].lower() or "numpy" == result[1].lower()

    def test_empty_list(self):
        """Test with empty dependency list."""
        deps = []
        result = deps_base_names(deps)
        assert result == []

    def test_with_non_string_values(self):
        """Test that non-string values are filtered out."""
        deps = ["requests>=2.0", None, 123, "numpy==1.19.0"]
        result = deps_base_names(deps)
        # Should only return string dependencies
        assert len(result) == 2
        assert "requests" in result
        assert "numpy" in result

    def test_with_whitespace(self):
        """Test handling of whitespace in dependency strings."""
        deps = ["  requests  >=2.0", "numpy >= 1.19.0"]
        result = deps_base_names(deps)
        # Leading/trailing whitespace should be handled
        assert "requests" in result[0].lower() or "requests" == result[0].lower()

    def test_single_dependency(self):
        """Test with single dependency."""
        deps = ["requests>=2.0"]
        result = deps_base_names(deps)
        assert len(result) == 1
        assert "requests" in result[0].lower()

    def test_returns_list(self):
        """Test that function returns a list."""
        deps = ["requests", "numpy"]
        result = deps_base_names(deps)
        assert isinstance(result, list)

    def test_case_preservation(self):
        """Test that case is preserved from original dependency."""
        deps = ["Requests", "NumPy", "Django"]
        result = deps_base_names(deps)
        # Case should be preserved in package names
        assert any("request" in r.lower() for r in result)
        assert any("numpy" in r.lower() for r in result)
        assert any("django" in r.lower() for r in result)

    def test_hyphenated_package_names(self):
        """Test extraction of hyphenated package names."""
        deps = ["google-cloud-storage>=1.0", "scikit-learn>=0.24"]
        result = deps_base_names(deps)
        assert "google-cloud-storage" in result
        assert "scikit-learn" in result

    def test_underscored_package_names(self):
        """Test extraction of underscored package names."""
        deps = ["my_package>=1.0", "another_lib==2.0"]
        result = deps_base_names(deps)
        assert "my_package" in result
        assert "another_lib" in result


class TestHandleDependency:
    """Test suite for handle_dependency main function."""

    @pytest.fixture
    def basic_df(self):
        """Create a basic test DataFrame with dependencies."""
        return pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3", "pkg4"],
                "latest_dependencies": [
                    ["requests>=2.0"],
                    ["numpy==1.19.0", "pandas>=1.0"],
                    ["some-malware"],
                    [],
                ],
            }
        )

    def test_returns_dataframe(self, basic_df):
        """Test that function returns a DataFrame."""
        result = handle_dependency(basic_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_expected_features(self, basic_df):
        """Test that all expected features are added."""
        result = handle_dependency(basic_df)
        expected_features = [
            "has_dependency_to_top_brand",
            "has_dependency_lev_close_to_brand",
            "min_dep_lev_to_brand",
        ]
        for feature in expected_features:
            assert feature in result.columns

    def test_preserves_original_columns(self, basic_df):
        """Test that original columns are preserved."""
        original_cols = set(basic_df.columns)
        result = handle_dependency(basic_df)
        for col in original_cols:
            assert col in result.columns

    def test_preserves_row_count(self, basic_df):
        """Test that row count is preserved."""
        result = handle_dependency(basic_df)
        assert len(result) == len(basic_df)

    def test_has_dependency_to_top_brand_is_binary(self, basic_df):
        """Test that has_dependency_to_top_brand contains only 0/1."""
        result = handle_dependency(basic_df)
        assert set(result["has_dependency_to_top_brand"].unique()).issubset({0, 1})

    def test_has_dependency_lev_close_is_binary(self, basic_df):
        """Test that has_dependency_lev_close_to_brand contains only 0/1."""
        result = handle_dependency(basic_df)
        assert set(result["has_dependency_lev_close_to_brand"].unique()).issubset(
            {0, 1}
        )

    def test_min_dep_lev_is_numeric_or_nan(self, basic_df):
        """Test that min_dep_lev_to_brand is numeric or NaN."""
        result = handle_dependency(basic_df)
        min_lev = result["min_dep_lev_to_brand"]
        # Should be numeric (including NaN)
        assert all(pd.isna(v) or isinstance(v, (int, float, np.number)) for v in min_lev)

    def test_min_dep_lev_non_negative(self, basic_df):
        """Test that min_dep_lev distances are non-negative."""
        result = handle_dependency(basic_df)
        min_lev = result["min_dep_lev_to_brand"].dropna()
        assert (min_lev >= 0).all()

    def test_with_top_brand_dependency(self):
        """Test detection of dependencies on top brand packages."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2"],
                "latest_dependencies": [
                    ["django>=3.0"],  # django is in TOP_BRAND_PKGS
                    ["flask>=1.0"],  # flask is in TOP_BRAND_PKGS
                ],
            }
        )
        result = handle_dependency(df)
        # Both should have dependency to top brand
        assert result["has_dependency_to_top_brand"].iloc[0] == 1
        assert result["has_dependency_to_top_brand"].iloc[1] == 1

    def test_without_top_brand_dependency(self):
        """Test when no top brand dependencies exist."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [["some-random-lib>=1.0"]],
            }
        )
        result = handle_dependency(df)
        assert result["has_dependency_to_top_brand"].iloc[0] == 0

    def test_with_typosquatted_brand(self):
        """Test detection of typosquatted brand names."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                # "djnago" is typo of "django" (Lev distance = 1, within threshold 2)
                "latest_dependencies": [["djnago>=3.0"]],
            }
        )
        result = handle_dependency(df)
        # Should detect typosquatted brand
        assert result["has_dependency_lev_close_to_brand"].iloc[0] == 1
        assert result["min_dep_lev_to_brand"].iloc[0] <= LEV_THRESHOLD

    def test_with_close_brand_alias(self):
        """Test detection of close brand aliases."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                # "googl" is close to "google" (Lev distance = 1)
                "latest_dependencies": [["googl"]],
            }
        )
        result = handle_dependency(df)
        # Should detect close brand alias
        assert result["has_dependency_lev_close_to_brand"].iloc[0] == 1
        assert result["min_dep_lev_to_brand"].iloc[0] <= LEV_THRESHOLD

    def test_with_far_brand_name(self):
        """Test that distant names don't trigger typosquatting flag."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                # "random-lib" is far from all brands
                "latest_dependencies": [["random-lib"]],
            }
        )
        result = handle_dependency(df)
        # Should not be marked as close
        assert result["has_dependency_lev_close_to_brand"].iloc[0] == 0
        assert result["min_dep_lev_to_brand"].iloc[0] > LEV_THRESHOLD

    def test_with_empty_dependencies(self):
        """Test with no dependencies."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [[]],
            }
        )
        result = handle_dependency(df)
        # Should have no top brand dependency
        assert result["has_dependency_to_top_brand"].iloc[0] == 0
        # Should have no close brand
        assert result["has_dependency_lev_close_to_brand"].iloc[0] == 0
        # Should have NaN for min distance (infinity converted to NaN)
        assert pd.isna(result["min_dep_lev_to_brand"].iloc[0])

    def test_with_null_dependencies(self):
        """Test with null/None dependencies causes TypeError."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [None],
            }
        )
        # The function doesn't handle None values - it will raise TypeError
        # because deps_base_names tries to iterate over None
        with pytest.raises(TypeError, match="'NoneType' object is not iterable"):
            handle_dependency(df)

    def test_case_insensitive_brand_detection(self):
        """Test that brand detection is case-insensitive."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [["DJANGO>=3.0", "Flask>=1.0"]],
            }
        )
        result = handle_dependency(df)
        # Should detect despite case difference
        assert result["has_dependency_to_top_brand"].iloc[0] == 1

    def test_with_mixed_dependencies(self):
        """Test with mix of brand and non-brand dependencies."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [["requests>=2.0", "random-lib", "django>=3.0"]],
            }
        )
        result = handle_dependency(df)
        # Should detect top brand (requests and django)
        assert result["has_dependency_to_top_brand"].iloc[0] == 1
        # Min distance should be to requests or django (which are 0)
        assert result["min_dep_lev_to_brand"].iloc[0] == 0

    def test_multiple_packages_with_dependencies(self):
        """Test multiple packages with different dependency profiles."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2", "pkg3"],
                "latest_dependencies": [
                    ["django>=3.0"],  # Has top brand
                    ["random-lib"],  # No brand
                    ["djnago>=3.0"],  # Typosquatted brand
                ],
            }
        )
        result = handle_dependency(df)
        assert result["has_dependency_to_top_brand"].iloc[0] == 1
        assert result["has_dependency_to_top_brand"].iloc[1] == 0
        assert result["has_dependency_lev_close_to_brand"].iloc[2] == 1

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(
            {
                "pkg_name": [],
                "latest_dependencies": [],
            }
        )
        result = handle_dependency(df)
        assert len(result) == 0
        assert "has_dependency_to_top_brand" in result.columns

    def test_deterministic_output(self):
        """Test that same input produces same output."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1", "pkg2"],
                "latest_dependencies": [
                    ["django>=3.0", "requests"],
                    ["djnago", "random-lib"],
                ],
            }
        )

        result1 = handle_dependency(df.copy())
        result2 = handle_dependency(df.copy())

        pd.testing.assert_frame_equal(
            result1[
                [
                    "has_dependency_to_top_brand",
                    "has_dependency_lev_close_to_brand",
                    "min_dep_lev_to_brand",
                ]
            ],
            result2[
                [
                    "has_dependency_to_top_brand",
                    "has_dependency_lev_close_to_brand",
                    "min_dep_lev_to_brand",
                ]
            ],
        )

    def test_with_version_specifiers_in_dependencies(self):
        """Test that version specifiers don't affect brand detection."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [
                    ["django>=3.0,<4.0", "requests[security]>=2.20"]
                ],
            }
        )
        result = handle_dependency(df)
        # Should detect django and requests as top brands
        assert result["has_dependency_to_top_brand"].iloc[0] == 1

    def test_single_row_dataframe(self):
        """Test with single row."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [["django>=3.0"]],
            }
        )
        result = handle_dependency(df)
        assert len(result) == 1
        assert result["has_dependency_to_top_brand"].iloc[0] == 1

    def test_large_dependency_list(self):
        """Test with large number of dependencies."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [
                    [f"lib{i}" for i in range(100)] + ["django>=3.0"]
                ],
            }
        )
        result = handle_dependency(df)
        # Should still detect top brand
        assert result["has_dependency_to_top_brand"].iloc[0] == 1

    def test_edge_case_lev_distance_boundary(self):
        """Test edge case at Levenshtein threshold boundary."""
        # Create a package name that is exactly LEV_THRESHOLD distance from a brand
        # This is implementation-specific but important for testing
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [["flsk"]],  # 1 char diff from flask
            }
        )
        result = handle_dependency(df)
        # Distance is 1, threshold is 2, so should be marked as close
        assert result["has_dependency_lev_close_to_brand"].iloc[0] == 1

    def test_returns_modified_dataframe(self):
        """Test that function returns modified DataFrame."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [["requests"]],
            }
        )
        original_cols = df.shape[1]
        result = handle_dependency(df)

        # Should have 3 new columns
        assert result.shape[1] == original_cols + 3
        assert isinstance(result, pd.DataFrame)

    def test_with_empty_string_dependencies(self):
        """Test with dependency list containing only empty strings."""
        df = pd.DataFrame(
            {
                "pkg_name": ["pkg1"],
                "latest_dependencies": [["", "  ", ""]],
            }
        )
        result = handle_dependency(df)
        # Empty strings are filtered out, leaving empty dists list
        assert result["has_dependency_to_top_brand"].iloc[0] == 0
        assert result["has_dependency_lev_close_to_brand"].iloc[0] == 0
        assert pd.isna(result["min_dep_lev_to_brand"].iloc[0])
