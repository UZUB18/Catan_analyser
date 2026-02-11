import unittest

from catan_analyzer.analysis.types import AnalysisMode
from catan_analyzer.ui.mode_descriptions import get_mode_description


class ModeDescriptionTests(unittest.TestCase):
    def test_all_known_modes_have_ui_metadata_fields(self) -> None:
        for mode in AnalysisMode:
            text = get_mode_description(mode).to_ui_text()
            self.assertIn("Reliability:", text)
            self.assertIn("Speed:", text)
            self.assertIn("Best use:", text)

    def test_hybrid_alias_strings_use_hybrid_copy(self) -> None:
        description = get_mode_description("hybrid_opening")
        self.assertIn("Hybrid strategy", description.summary)
        self.assertIn("balanced pick quality", description.reliability)

    def test_unknown_mode_falls_back_to_generic_copy(self) -> None:
        description = get_mode_description("experimental_mode")
        self.assertEqual(description.summary, "Custom analysis mode.")


if __name__ == "__main__":
    unittest.main()
