import unittest

from core.profile_updater import ProfileUpdater


class TestProfileUpdater(unittest.TestCase):
    def setUp(self) -> None:
        self.updater = ProfileUpdater()

    def test_new_fields_added_without_conflict(self) -> None:
        merged, conflicts = self.updater.merge({}, {"annual_income": 90000})
        self.assertEqual(merged["annual_income"], 90000)
        self.assertEqual(conflicts, [])

    def test_same_value_again_is_not_a_conflict(self) -> None:
        merged, conflicts = self.updater.merge(
            {"annual_income": 90000}, {"annual_income": 90000}
        )
        self.assertEqual(conflicts, [])

    def test_different_value_is_flagged_as_conflict(self) -> None:
        merged, conflicts = self.updater.merge(
            {"annual_income": 90000}, {"annual_income": 120000}
        )
        self.assertEqual(len(conflicts), 1)
        self.assertIn("annual_income", conflicts[0])

    def test_conflicting_value_still_overwrites_profile(self) -> None:
        merged, conflicts = self.updater.merge(
            {"credit_score": 600}, {"credit_score": 720}
        )
        self.assertEqual(merged["credit_score"], 720)

    def test_existing_unrelated_fields_are_preserved(self) -> None:
        merged, conflicts = self.updater.merge(
            {"annual_income": 90000}, {"credit_score": 700}
        )
        self.assertEqual(merged["annual_income"], 90000)
        self.assertEqual(merged["credit_score"], 700)


if __name__ == "__main__":
    unittest.main()