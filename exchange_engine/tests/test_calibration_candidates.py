import unittest

from src.backtest import BacktestConfiguration
from src.calibration_candidates import (
    CANDIDATE_MANIFEST_COLUMNS,
    CandidatePlan,
    CandidateConfigurationSpec,
    PolicyCandidate,
    ThresholdCandidate,
    build_candidate_plan,
    build_initial_candidate_plan,
    initial_configuration_specs,
    initial_policy_candidates,
    initial_threshold_candidates,
)
from src.signal_engine import SignalDecisionPolicy, SignalThresholds


def sample_thresholds() -> SignalThresholds:
    return SignalThresholds(
        sma60_good=-2.0,
        sma60_strong=-4.0,
        sma120_good=-2.0,
        sma120_strong=-4.0,
        percentile_good=25.0,
        percentile_strong=10.0,
        bollinger_near_lower_pct=1.0,
    )


def sample_policy() -> SignalDecisionPolicy:
    return SignalDecisionPolicy(
        minimum_available_conditions=3,
        watch_min_satisfied_conditions=1,
        good_min_satisfied_conditions=3,
        strong_min_satisfied_conditions=4,
        strong_min_strong_conditions=3,
    )


class InitialCandidatePlanTests(unittest.TestCase):
    def test_initial_plan_is_explicit_bounded_and_stably_ordered(self) -> None:
        plan = build_initial_candidate_plan()

        self.assertEqual(len(plan.threshold_candidates), 10)
        self.assertEqual(len(plan.policy_candidates), 3)
        self.assertEqual(plan.candidate_count, 12)
        self.assertLessEqual(plan.candidate_count, plan.max_candidate_count)
        self.assertEqual(
            [item.configuration_id for item in plan.configurations],
            [
                "baseline__conservative",
                "baseline__balanced",
                "baseline__sensitive",
                "sma60_sensitive__balanced",
                "sma60_strict__balanced",
                "sma120_sensitive__balanced",
                "sma120_strict__balanced",
                "percentile_sensitive__balanced",
                "percentile_strict__balanced",
                "bollinger_strict__balanced",
                "bollinger_tight__balanced",
                "bollinger_sensitive__balanced",
            ],
        )
        self.assertTrue(
            all(
                isinstance(configuration, BacktestConfiguration)
                for configuration in plan.configurations
            )
        )

    def test_threshold_profiles_cover_documented_exploratory_values(self) -> None:
        candidates = {
            item.threshold_id: item.thresholds
            for item in initial_threshold_candidates()
        }

        self.assertEqual(candidates["baseline"].sma60_good, -2.0)
        self.assertEqual(candidates["sma60_sensitive"].sma60_good, -1.0)
        self.assertEqual(candidates["sma60_strict"].sma60_good, -3.0)
        self.assertEqual(candidates["sma120_sensitive"].sma120_strong, -2.0)
        self.assertEqual(candidates["sma120_strict"].sma120_strong, -5.0)
        self.assertEqual(candidates["percentile_strict"].percentile_good, 20.0)
        self.assertEqual(
            candidates["percentile_sensitive"].percentile_strong,
            15.0,
        )
        self.assertEqual(candidates["bollinger_strict"].bollinger_near_lower_pct, 0.0)
        self.assertEqual(candidates["bollinger_tight"].bollinger_near_lower_pct, 0.5)
        self.assertEqual(
            candidates["bollinger_sensitive"].bollinger_near_lower_pct,
            2.0,
        )
        for thresholds in candidates.values():
            self.assertLessEqual(thresholds.sma60_strong, thresholds.sma60_good)
            self.assertLessEqual(thresholds.sma120_strong, thresholds.sma120_good)
            self.assertLessEqual(
                thresholds.percentile_strong,
                thresholds.percentile_good,
            )

    def test_policy_profiles_have_distinct_valid_meanings(self) -> None:
        policies = {
            item.policy_id: item.policy for item in initial_policy_candidates()
        }

        self.assertEqual(
            policies["conservative"],
            SignalDecisionPolicy(4, 2, 4, 4, 4),
        )
        self.assertEqual(
            policies["balanced"],
            SignalDecisionPolicy(3, 1, 3, 4, 3),
        )
        self.assertEqual(
            policies["sensitive"],
            SignalDecisionPolicy(2, 1, 2, 3, 2),
        )

    def test_manifest_preserves_every_threshold_and_policy_field(self) -> None:
        plan = build_initial_candidate_plan()
        manifest = plan.to_manifest()

        self.assertEqual(manifest.columns.tolist(), list(CANDIDATE_MANIFEST_COLUMNS))
        self.assertEqual(len(manifest), plan.candidate_count)
        baseline = manifest.loc[
            manifest["configuration_id"].eq("baseline__balanced")
        ].iloc[0]
        self.assertEqual(baseline["threshold_id"], "baseline")
        self.assertEqual(baseline["policy_id"], "balanced")
        self.assertEqual(baseline["sma60_good"], -2.0)
        self.assertEqual(baseline["strong_min_strong_conditions"], 3)
        self.assertFalse(
            any(
                word in column.lower()
                for column in manifest.columns
                for word in ("winner", "rank", "score", "best")
            )
        )


class CandidatePlanValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.threshold = ThresholdCandidate("threshold", sample_thresholds())
        self.policy = PolicyCandidate("policy", sample_policy())
        self.spec = CandidateConfigurationSpec(
            "threshold__policy",
            "threshold",
            "policy",
        )

    def build(self, **overrides: object):
        values = {
            "threshold_candidates": (self.threshold,),
            "policy_candidates": (self.policy,),
            "specifications": (self.spec,),
            "max_candidate_count": 1,
        }
        values.update(overrides)
        return build_candidate_plan(**values)

    def test_candidate_count_limit_fails_before_backtest(self) -> None:
        self.assertEqual(self.build().candidate_count, 1)
        with self.assertRaisesRegex(
            ValueError,
            "candidate configuration count 12 exceeds max_candidate_count 11",
        ):
            build_initial_candidate_plan(max_candidate_count=11)

    def test_rejects_duplicate_ids_pairs_and_unknown_references(self) -> None:
        cases = (
            (
                {
                    "threshold_candidates": (
                        self.threshold,
                        ThresholdCandidate("threshold", sample_thresholds()),
                    )
                },
                "threshold_id values must be unique",
            ),
            (
                {
                    "policy_candidates": (
                        self.policy,
                        PolicyCandidate("policy", sample_policy()),
                    )
                },
                "policy_id values must be unique",
            ),
            (
                {
                    "specifications": (
                        self.spec,
                        CandidateConfigurationSpec(
                            "threshold__policy",
                            "threshold",
                            "policy",
                        ),
                    ),
                    "max_candidate_count": 2,
                },
                "configuration_id values must be unique",
            ),
            (
                {
                    "specifications": (
                        self.spec,
                        CandidateConfigurationSpec(
                            "same_pair",
                            "threshold",
                            "policy",
                        ),
                    ),
                    "max_candidate_count": 2,
                },
                "pairs must be unique",
            ),
            (
                {
                    "specifications": (
                        CandidateConfigurationSpec(
                            "unknown_threshold",
                            "missing",
                            "policy",
                        ),
                    )
                },
                "unknown threshold_id",
            ),
            (
                {
                    "specifications": (
                        CandidateConfigurationSpec(
                            "unknown_policy",
                            "threshold",
                            "missing",
                        ),
                    )
                },
                "unknown policy_id",
            ),
        )
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.build(**overrides)

    def test_rejects_empty_malformed_and_invalid_limit_inputs(self) -> None:
        for field_name in (
            "threshold_candidates",
            "policy_candidates",
            "specifications",
        ):
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(ValueError, "must not be empty"):
                    self.build(**{field_name: ()})

        for maximum, expected_error in (
            (0, ValueError),
            (-1, ValueError),
            (True, TypeError),
            (1.0, TypeError),
        ):
            with self.subTest(maximum=maximum):
                with self.assertRaises(expected_error):
                    self.build(max_candidate_count=maximum)

        with self.assertRaisesRegex(ValueError, "must match"):
            ThresholdCandidate("Invalid ID", sample_thresholds())
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            CandidatePlan(
                threshold_candidates=(self.threshold,),
                policy_candidates=(self.policy,),
                specifications=(self.spec,),
                configurations=(),
                max_candidate_count=1,
            )
        with self.assertRaises(ValueError):
            SignalThresholds(
                sma60_good=-3.0,
                sma60_strong=-2.0,
                sma120_good=-2.0,
                sma120_strong=-4.0,
                percentile_good=25.0,
                percentile_strong=10.0,
                bollinger_near_lower_pct=1.0,
            )


if __name__ == "__main__":
    unittest.main()
