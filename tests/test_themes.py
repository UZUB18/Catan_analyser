import unittest

from catan_analyzer.ui.themes import available_themes, get_theme


class ThemeTests(unittest.TestCase):
    def test_expected_themes_are_available(self) -> None:
        keys = set(available_themes())
        self.assertTrue({"light", "dark", "high_contrast"}.issubset(keys))

    def test_unknown_theme_falls_back_to_light(self) -> None:
        theme = get_theme("unknown_theme_key")
        self.assertEqual(theme.key, "light")


if __name__ == "__main__":
    unittest.main()
