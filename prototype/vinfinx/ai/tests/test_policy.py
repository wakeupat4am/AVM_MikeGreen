"""The policy engine (docs/POLICY_ENGINE_PLAN.md): reproduces the frozen cases' hand-written policy results, and
applies each rule of the provisional policy config P1-P5."""
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import yaml

from ai.evaluation import runner
from ai.evaluation.cases import DecisionInput, build_bundle, load_case_set
from ai.policy import POLICY_PATH, PolicyConfig, compute_policy, load_policy_config
from ai.policy.provisional_mapping import provisional_mapping_set
from ai.schemas.mapping import MappingSet
from ai.tests.fixtures import CAT, NOW, item

CASES_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "cases" / "adversarial_v0.yaml"
SEPT = (datetime(2026, 9, 1, tzinfo=timezone.utc), datetime(2026, 9, 30, 23, 59, 59, tzinfo=timezone.utc))
OCT1 = (datetime(2026, 10, 1, tzinfo=timezone.utc), datetime(2026, 10, 1, 23, 59, 59, tzinfo=timezone.utc))


def observed(metric_id, value, baseline, period=SEPT, label="2026-09", unit="pct", **changes):
    base = item(metric_id, label, *period, value or "0", unit, "level")
    data = {**base.model_dump(), "value": value, **changes}
    if baseline is not None:
        data["baseline"] = {"value": baseline, "unit": unit, "period_label": "base"}
    return type(base).model_validate(data)


def refi(value="4.0", baseline="4.5", **changes):
    return observed("macro.vn.refi_rate", value, baseline, **changes)


def mortgage(value="9.2", baseline="9.8"):
    return observed("macro.vn.mortgage_rate_avg", value, baseline)


def inventory(district="hn-dong-da", value="1250", baseline="1100"):
    return observed(f"market.{district}.listing_inventory", value, baseline, period=OCT1, label="2026-10-01",
                    unit="count")


def config_with(**changes) -> PolicyConfig:
    with open(POLICY_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw.update(changes)
    return PolicyConfig.model_validate(raw)


def validated_mapping() -> MappingSet:
    m = provisional_mapping_set()
    return MappingSet.model_validate({**m.model_dump(), "status": "validated", "reviewed_by": "reviewer",
                                      "reviewed_at": NOW})


class EngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_policy_config(catalogue=CAT)
        cls.mapping = provisional_mapping_set()

    def policy(self, items, *, mapping=None, config=None, district="hn-dong-da", now=NOW):
        return compute_policy(items, mapping or self.mapping, config or self.config, catalogue=CAT,
                              customer_district=district, case_id="case-1", now=now)

    def target(self, items, **kw):
        (t,) = self.policy(items, **kw).targets
        return t

    def test_single_signal_raises(self):
        t = self.target([refi()])
        self.assertEqual((t.outcome, t.net_weight, t.magnitude_cap.value), ("raises", Decimal("1.0"), "low"))
        self.assertEqual([a.value for a in t.allowed_actions], ["offer_review", "request_evidence", "no_change"])
        self.assertEqual(t.signals[0].effect, "raises")  # a rate cut, direction negative

    def test_effect_follows_change_and_direction(self):
        self.assertEqual(self.target([refi("5.0", "4.5")]).outcome, "lowers")  # a rate rise
        self.assertEqual(self.target([inventory()]).signals[0].effect, "lowers")  # more listings
        self.assertEqual(self.target([inventory(value="1000")]).signals[0].effect, "raises")

    def test_conflict_and_dominant_direction(self):
        t = self.target([refi(), inventory()])  # +1.0 - 0.9 = 0.1
        self.assertEqual((t.outcome, t.must_abstain, t.magnitude_cap.value), ("conflict", True, "none"))
        self.assertEqual([a.value for a in t.allowed_actions], ["abstain", "request_evidence"])
        # at the threshold the net direction wins
        self.assertEqual(self.target([refi(), inventory()], config=config_with(conflict_threshold="0.1")).outcome,
                         "raises")
        # agreeing signals add up
        self.assertEqual(self.target([refi(), mortgage()]).net_weight, Decimal("1.8"))

    def test_exclusions(self):
        stale = self.target([refi()], now=refi().freshness.deadline + timedelta(days=1)).signals[0]
        self.assertEqual((stale.material, stale.excluded_reason, stale.effect), (False, "STALE", "raises"))
        unreadable = refi(value=None, presence="unreadable", trust="untrusted", trust_reasons=["VALUE_UNREADABLE"])
        s = self.target([unreadable]).signals[0]
        self.assertEqual((s.excluded_reason, s.effect), ("UNTRUSTED", None))
        self.assertEqual(self.target([refi(baseline=None)]).signals[0].excluded_reason, "NO_BASELINE")
        small = self.target([refi("4.4", "4.5")])  # 0.1 pp < 0.25 pp
        self.assertEqual((small.signals[0].excluded_reason, small.outcome), ("BELOW_MATERIALITY", "none"))
        self.assertEqual([a.value for a in small.allowed_actions], ["request_evidence", "no_change", "abstain"])
        # relative threshold: 5% of 1100 is 55
        self.assertEqual(self.target([inventory(value="1150")]).signals[0].excluded_reason, "BELOW_MATERIALITY")
        self.assertTrue(self.target([inventory(value="1160")]).signals[0].material)

    def test_only_the_customers_district_and_weighted_metrics(self):
        t = self.target([refi(), inventory("hn-ba-dinh"), observed("macro.vn.unemployment_rate", "2.2", "2.3")])
        self.assertEqual([s.metric_key for s in t.signals], ["refi_rate"])  # ADV-029, ADV-028
        t = self.target([refi(), inventory("hn-ba-dinh")], district="hn-ba-dinh")
        self.assertEqual(t.outcome, "conflict")

    def test_validated_mapping(self):
        t = self.target([refi(), mortgage()], mapping=validated_mapping())  # net 1.8 >= 1.5
        self.assertEqual(t.magnitude_cap.value, "medium")
        self.assertEqual(self.target([refi()], mapping=validated_mapping()).magnitude_cap.value, "low")
        lowers = self.target([refi("5.0", "4.5")], mapping=validated_mapping())
        self.assertIn("hold", [a.value for a in lowers.allowed_actions])
        # unvalidated: never hold, never above low
        self.assertNotIn("hold", [a.value for a in self.target([refi("5.0", "4.5")]).allowed_actions])
        self.assertEqual(self.target([refi(), mortgage()]).magnitude_cap.value, "low")

    def test_rejected_mapping_is_never_applied(self):
        m = MappingSet.model_validate({**self.mapping.model_dump(), "status": "rejected", "reviewed_by": "r",
                                       "reviewed_at": NOW})
        with self.assertRaises(ValueError):
            self.policy([refi()], mapping=m)

    def test_result_is_versioned(self):
        p = self.policy([refi()])
        self.assertEqual((p.policy_version, p.policy_hash), ("policy.v0", self.config.policy_hash))
        self.assertEqual(p.mapping_id, self.mapping.mapping_id)
        self.assertNotEqual(config_with(conflict_threshold="0.6").policy_hash, self.config.policy_hash)
        self.assertEqual(provisional_mapping_set().mapping_id, self.mapping.mapping_id)  # stable


class ConfigTest(unittest.TestCase):
    def test_provisional_and_checked(self):
        config = load_policy_config(catalogue=CAT)
        self.assertEqual((config.status, config.signed_off_by), ("provisional", None))
        with self.assertRaises(ValueError):
            config_with(status="approved")  # approval needs the domain owner's sign-off
        with self.assertRaises(ValueError):
            config_with(materiality={"refi_rate": {"absolute": "0.25"}})  # weighted metrics without thresholds
        with self.assertRaises(ValueError):
            config_with(weights={"borrowing_capacity": {"gdp": "1.0"}}, materiality={"gdp": {"absolute": "1"}}
                        ).check_against(CAT)


class ReproducesFrozenCasesTest(unittest.TestCase):
    """Acceptance test 1: the engine gives the policy result each case author wrote. A consistency check of the
    hand-written inputs (all splits), not a score."""

    def test_every_decision_case(self):
        engine = runner.EnginePolicy(CAT)
        cases = load_case_set(CASES_PATH)
        compared = 0
        for case in cases.cases:
            if not isinstance(case.input, DecisionInput):
                continue
            with self.subTest(case.case_id):
                hand = build_bundle(case, CAT).policy
                computed = build_bundle(case, CAT, engine.for_case(case)).policy
                self.assertEqual(computed.mapping_status, hand.mapping_status)
                for h, c in zip(hand.targets, computed.targets, strict=True):
                    hd, cd = h.model_dump(), c.model_dump()
                    for hs, cs in zip(hd.pop("signals"), cd.pop("signals"), strict=True):
                        if cs["effect"] is None:
                            # Known difference: the case author gave an unreadable value an effect ("raises");
                            # the engine cannot compute one. The signal is excluded either way (ADV-014, 015).
                            self.assertFalse(hs["material"])
                            hs["effect"] = None
                        self.assertEqual(cs, hs)
                    self.assertEqual(cd, hd)
                compared += 1
        self.assertEqual(compared, 30)

    def test_runner_records_the_engine(self):
        report = runner.run(CASES_PATH, "scripted", "all", runner.DEFAULT_URL, None, policy="engine")
        self.assertEqual(report["policy"]["source"], "engine")
        self.assertEqual(report["policy"]["policy_version"], "policy.v0")
        ran = [c for c in report["cases"] if not c["skipped"]]
        self.assertEqual([c["case_id"] for c in ran if not c["passed"]], [])


if __name__ == "__main__":
    unittest.main()
