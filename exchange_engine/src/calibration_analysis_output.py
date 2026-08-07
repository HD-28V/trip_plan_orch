"""Output helpers for unranked calibration review artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


SHORTLIST_COLUMNS = (
    "configuration_id",
    "threshold_id",
    "policy_id",
    "analysis_group",
    "evidence_axis",
    "period_stability_axis",
    "currency_stability_axis",
    "horizon_stability_axis",
    "signal_separation_axis",
    "validation_good_strong_sample_count",
    "validation_good_strong_occurrence_ratio",
    "validation_good_strong_evaluable_label_count",
    "validation_supported_cell_count",
    "calibration_validation_common_supported_cell_count",
    "validation_good_strong_favorable_ratio",
    "validation_good_strong_mean_advantage_pct",
    "validation_good_strong_median_advantage_pct",
    "matched_calibration_good_strong_favorable_ratio",
    "matched_validation_good_strong_favorable_ratio",
    "matched_calibration_good_strong_mean_advantage_pct",
    "matched_validation_good_strong_mean_advantage_pct",
    "matched_calibration_good_strong_median_advantage_pct",
    "matched_validation_good_strong_median_advantage_pct",
    "validation_minimum_coverage_ratio",
    "validation_supported_currency_count",
    "validation_positive_currency_count",
    "validation_minimum_currency_pooled_favorable_ratio",
    "validation_currency_pooled_favorable_stddev",
    "validation_currency_pooled_favorable_range",
    "validation_minimum_currency_good_strong_sample_count",
    "validation_horizon_comparable_currency_count",
    "validation_maximum_short_to_medium_favorable_drop",
    "validation_strong_sample_count",
    "validation_minimum_strong_currency_count",
    "favorable_ratio_gap",
    "mean_advantage_gap_pct",
    "median_advantage_gap_pct",
    "core_risk_axis_count",
    "risk_flags",
    "review_reasons",
    "human_decision_required",
)

OUTPUT_FILES = (
    ("analysis_metadata.csv", "analysis_metadata"),
    ("configuration_review.csv", "configuration_review"),
    ("stability_review.csv", "stability_review"),
    ("currency_stability_review.csv", "currency_stability_review"),
    ("horizon_stability_review.csv", "horizon_stability_review"),
    ("signal_separation_review.csv", "signal_separation_review"),
    ("candidate_shortlist.csv", "candidate_shortlist"),
)


def build_candidate_shortlist(
    configuration_review: pd.DataFrame,
) -> pd.DataFrame:
    """Return every PROMISING row in manifest order without ranking or filling."""

    required = set(SHORTLIST_COLUMNS).difference({"human_decision_required"})
    if not isinstance(configuration_review, pd.DataFrame):
        raise TypeError("configuration_review must be a pandas DataFrame")
    if not required.issubset(configuration_review.columns):
        raise ValueError("configuration_review is missing shortlist columns")

    shortlist = configuration_review.loc[
        configuration_review["analysis_group"].eq("PROMISING"),
        list(required),
    ].copy(deep=True)
    # Restore a deterministic, human-oriented schema rather than set order.
    shortlist = shortlist.loc[
        :,
        [
            column
            for column in SHORTLIST_COLUMNS
            if column != "human_decision_required"
        ],
    ]
    shortlist["human_decision_required"] = True
    return shortlist.loc[:, SHORTLIST_COLUMNS].reset_index(drop=True)


def _shortlist_status(count: int, heuristics: object) -> str:
    minimum = int(getattr(heuristics, "shortlist_min_count"))
    maximum = int(getattr(heuristics, "shortlist_max_count"))
    if minimum <= count <= maximum:
        return f"READY_FOR_HUMAN_REVIEW_{count}_UNRANKED"
    return f"SHORTLIST_SIZE_REQUIRES_HUMAN_REVIEW_{count}"


def build_analysis_metadata(
    *,
    heuristics: object,
    shortlist_status: str,
    shortlist_count: int,
) -> pd.DataFrame:
    """Document every analysis-only convention and heuristic value."""

    rows: list[tuple[str, object]] = [
        ("report_purpose", "human review of existing calibration candidates"),
        ("analysis_only", True),
        ("production_rule", False),
        ("winner_selected", False),
        ("ranking_or_score", "none"),
        ("candidate_count", 12),
        ("periods", "CALIBRATION,VALIDATION"),
        ("currencies", "USD,JPY,EUR"),
        ("horizons", "5,10,20,60"),
        ("signals", "WAIT,WATCH,GOOD,STRONG"),
        (
            "occurrence_sample_definition",
            "sum once per currency; never sum repeated horizons",
        ),
        (
            "evaluable_label_definition",
            "currency x signal x horizon non-NaN forward outcomes",
        ),
        (
            "compact_pooled_favorable_definition",
            "all evaluable GOOD+STRONG labels: sum favorable / sum evaluable",
        ),
        (
            "compact_weighted_mean_definition",
            "all evaluable GOOD+STRONG labels: evaluable-weighted cell means",
        ),
        (
            "compact_gap_definition",
            "VALIDATION - CALIBRATION on common supported currency x horizon cells",
        ),
        (
            "median_definition",
            "median of evaluable GOOD+STRONG cell medians; not a pooled raw median",
        ),
        ("gap_direction", "VALIDATION - CALIBRATION"),
        (
            "positive_cell_definition",
            "configured vote across favorable ratio, mean advantage, median advantage",
        ),
        (
            "period_shift_definition",
            "at least 2 of 3 absolute favorable/mean/median gaps are material",
        ),
        (
            "period_drift_definition",
            "at least 2 of 3 favorable/mean/median gaps materially deteriorate",
        ),
        (
            "currency_instability_definition",
            "3 supported currencies plus material pooled-favorable range and mean/median sign conflict",
        ),
        (
            "horizon_instability_definition",
            "supported short and medium evidence plus material favorable drop or positive-to-negative mean/median change",
        ),
        (
            "signal_separation_definition",
            "failure fraction among configured minimum comparable adjacent-signal cells",
        ),
        (
            "strong_too_rare_definition",
            "minimum currency STRONG occurrence is below very_low_sample_count",
        ),
        ("nan_policy", "unavailable evidence remains NaN; never zero-filled"),
        (
            "common_currency_policy",
            "common metrics require supported evidence in all three currencies",
        ),
        (
            "promising_group_policy",
            "3/3 supported and positive currencies; positive supported-cell ratio meets analysis heuristic; no validation drift or repeated STRONG<GOOD",
        ),
        (
            "weak_group_policy",
            "not PROMISING and at least two core risk axes, or supported positive-cell ratio below 0.50 with adequate evidence",
        ),
        (
            "group_risk_policy",
            "risk flags remain visible and do not by themselves remove a PROMISING review candidate",
        ),
        ("shortlist_policy", "all PROMISING rows, unranked, never fill or truncate"),
        ("shortlist_count", shortlist_count),
        ("shortlist_status", shortlist_status),
        (
            "period_length_warning",
            "raw CALIBRATION/VALIDATION sample-count gaps reflect unequal period lengths; compare occurrence ratios too",
        ),
        (
            "overlap_warning",
            "forward horizons overlap; evaluable labels are not independent samples",
        ),
    ]
    field_names = getattr(heuristics, "__dataclass_fields__", {})
    for field_name in field_names:
        rows.append((f"analysis_heuristic_{field_name}", getattr(heuristics, field_name)))
    return pd.DataFrame(rows, columns=("field", "value"))


def render_analysis_notes(
    configuration_review: pd.DataFrame,
    shortlist: pd.DataFrame,
    *,
    heuristics: object,
    shortlist_status: str,
) -> str:
    """Render an auditable Korean review note without declaring a winner."""

    group_counts = configuration_review["analysis_group"].value_counts(sort=False)
    lines = [
        "# Threshold Calibration 분석 메모",
        "",
        "이 문서는 기존 12개 configuration의 사람 검토를 돕는 분석 자료입니다.",
        "production threshold, winner, score 또는 순위를 만들지 않습니다.",
        "",
        "## 입력 검증",
        "",
        "- configuration 12개",
        "- CALIBRATION / VALIDATION",
        "- USD / JPY / EUR",
        "- horizon 5 / 10 / 20 / 60",
        "- WAIT / WATCH / GOOD / STRONG",
        "- count, ratio, NaN 및 cross-currency 산식 재검증",
        "",
        "## 집계 원칙",
        "",
        "- 동일 signal occurrence는 네 horizon에 반복되므로 sample count에 한 번만 포함합니다.",
        "- favorable ratio는 favorable count 합계를 evaluable count 합계로 나눕니다.",
        "- mean advantage는 evaluable count로 가중합니다.",
        "- raw outcome이 없으므로 compact median은 evaluable cell median들의 중앙값입니다.",
        "- headline CAL/VAL 값은 모든 evaluable GOOD+STRONG label을 사용합니다.",
        "- 성능 gap은 두 기간에 공통으로 지원되는 통화×horizon cell에서 VALIDATION - CALIBRATION으로 계산하며 signed/absolute 값을 함께 제공합니다.",
        "- 표본 미달 또는 분모 0은 0으로 바꾸지 않고 NaN/UNAVAILABLE로 유지합니다.",
        "- CAL/VAL 기간 길이가 다르므로 raw sample count gap은 occurrence ratio와 함께 봅니다.",
        "",
        "## 분석용 heuristic",
        "",
        "아래 값은 production 기준이 아니라 저표본과 불안정을 표시하기 위한 공개 heuristic입니다.",
        "",
        f"- LOW_SAMPLE: evaluable < {int(getattr(heuristics, 'low_sample_count'))}",
        f"- VERY_LOW_SAMPLE: evaluable < {int(getattr(heuristics, 'very_low_sample_count'))}",
        f"- LOW_COVERAGE: coverage < {float(getattr(heuristics, 'low_coverage_ratio')):.2f}",
        f"- SEVERE_OUTCOME_CENSORING: coverage < {float(getattr(heuristics, 'severe_coverage_ratio')):.2f}",
        f"- material favorable gap/range: {float(getattr(heuristics, 'material_favorable_gap')):.2f}",
        f"- material mean/median advantage gap: {float(getattr(heuristics, 'material_mean_gap_pct')):.2f} pct",
        f"- currency favorable range: {float(getattr(heuristics, 'material_currency_favorable_range')):.2f}",
        f"- horizon favorable drop: {float(getattr(heuristics, 'material_horizon_favorable_drop')):.2f}",
        f"- signal comparison minimum evaluable: {int(getattr(heuristics, 'signal_comparison_min_evaluable_count'))}",
        f"- signal comparison minimum comparable cells: {int(getattr(heuristics, 'minimum_comparable_separation_cells'))}",
        f"- signal comparison failure fraction: {float(getattr(heuristics, 'comparison_failure_fraction')):.2f}",
        f"- positive cell metric votes: {int(getattr(heuristics, 'positive_metric_vote_count'))}/3",
        f"- positive favorable floor: > {float(getattr(heuristics, 'positive_favorable_ratio_floor')):.2f}",
        f"- positive advantage floor: > {float(getattr(heuristics, 'positive_advantage_floor_pct')):.2f} pct",
        f"- positive currency supported-cell ratio: {float(getattr(heuristics, 'minimum_positive_currency_cell_ratio')):.2f}",
        f"- PROMISING positive supported-cell ratio: {float(getattr(heuristics, 'minimum_positive_supported_cell_ratio')):.2f}",
        f"- WEAK core risk-axis count: {int(getattr(heuristics, 'weak_core_risk_axis_count'))}",
        f"- WEAK positive supported-cell ratio: < {float(getattr(heuristics, 'weak_positive_supported_cell_ratio')):.2f}",
        "",
        "### Risk flag 산식",
        "",
        "- positive cell: favorable ratio, mean advantage, median advantage 중 설정된 vote 수 이상이 양수 방향입니다.",
        "- CALIBRATION_VALIDATION_SHIFT/DRIFT: favorable·mean·median gap 중 2개 이상이 material 기준을 넘으며, DRIFT는 악화 방향만 셉니다.",
        "- CURRENCY_INSTABILITY: 세 통화가 지원되고 pooled favorable range가 기준 이상이며 mean 또는 median에 부호 충돌이 있습니다.",
        "- HORIZON_INSTABILITY: short/medium evidence가 모두 있고 favorable이 material하게 하락하거나 mean/median이 양수에서 음수로 바뀝니다.",
        "- WEAK_SIGNAL_SEPARATION: 비교 가능한 인접 signal cell에서 WORSE/NOT_SEPARATED 비율이 설정 기준 이상입니다.",
        "- STRONG_NOT_BETTER_THAN_GOOD: 비교 가능한 GOOD→STRONG cell 중 WORSE 비율이 설정 기준 이상입니다.",
        "- STRONG_TOO_RARE: 통화별 STRONG occurrence 최솟값이 VERY_LOW_SAMPLE 기준보다 작습니다.",
        "- 위 flag는 자동 탈락이나 production rule이 아니며 상세 evidence를 찾기 위한 표시입니다.",
        "",
        "## 검토 그룹",
        "",
    ]
    for group in ("PROMISING", "MIXED", "WEAK"):
        lines.append(f"- {group}: {int(group_counts.get(group, 0))}")

    lines.extend(
        [
            "",
            "PROMISING은 세 통화 모두에 지원되는 긍정 evidence가 있고, 지원 cell의 긍정 비율이 분석용 기준을 충족하며, validation 붕괴나 반복적인 STRONG<GOOD가 없는 검토 그룹입니다.",
            "MIXED는 긍정 evidence와 저표본·불안정·비교 불가가 함께 있는 그룹입니다.",
            "WEAK은 PROMISING 조건을 충족하지 못하면서 여러 독립 risk 축이 반복되거나, 충분한 evidence의 긍정 cell 비율이 낮은 경우입니다.",
            "PROMISING에도 저표본·통화·horizon·separation risk flag는 그대로 남겨 사람 검토에서 함께 봅니다.",
            "표본 부족만으로 WEAK을 부여하지 않습니다.",
            "",
            "## 무순위 사람 검토 shortlist",
            "",
            f"- status: {shortlist_status}",
            f"- count: {len(shortlist)}",
        ]
    )
    if shortlist.empty:
        lines.append("- 후보 없음: 사람이 heuristic과 evidence를 다시 검토해야 합니다.")
    else:
        for row in shortlist.itertuples(index=False):
            lines.extend(
                [
                    "",
                    f"### {row.configuration_id}",
                    "",
                    f"- 남은 이유: {row.review_reasons}",
                    f"- 위험/제한: {row.risk_flags or 'NONE'}",
                ]
            )

    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- WAIT에는 indicator warm-up과 조건 미충족이 함께 섞여 있을 수 있습니다.",
            "- forward horizon은 서로 중첩되므로 독립 표본이나 독립 실험으로 해석할 수 없습니다.",
            "- 장기 horizon의 tail censoring 때문에 작은 표본의 100% favorable은 강한 증거가 아닙니다.",
            "- 통화·기간별 시장 국면 차이는 threshold 효과와 분리되지 않습니다.",
            "- 이 shortlist 내부에는 우선순위가 없으며 최종 선택은 사람이 수행해야 합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis_reports(
    report: object,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Write separate analysis artifacts while protecting every source CSV."""

    destination = Path(output_dir)
    source_dir = getattr(report, "source_dir", None)
    if source_dir is not None and destination.resolve() == Path(source_dir).resolve():
        raise ValueError("analysis output directory must differ from source directory")

    frames: list[tuple[str, pd.DataFrame]] = []
    for filename, attribute_name in OUTPUT_FILES:
        frame = getattr(report, attribute_name)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{attribute_name} must be a pandas DataFrame")
        frames.append((filename, frame))
    notes = getattr(report, "analysis_notes")
    if not isinstance(notes, str):
        raise TypeError("analysis_notes must be text")

    if destination.exists():
        raise FileExistsError(
            f"analysis output directory already exists: {destination.name}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(f".{destination.name}.pending")
    if pending.exists():
        raise FileExistsError(
            f"analysis staging directory already exists: {pending.name}"
        )

    csv_targets = tuple(destination / filename for filename, _ in OUTPUT_FILES)
    notes_target = destination / "analysis_notes.md"
    targets = (*csv_targets, notes_target)
    pending.mkdir()
    try:
        for filename, frame in frames:
            frame.to_csv(
                pending / filename,
                index=False,
                encoding="utf-8",
            )
        (pending / "analysis_notes.md").write_text(notes, encoding="utf-8")
        pending.replace(destination)
    except Exception:
        if pending.exists():
            shutil.rmtree(pending)
        raise
    return targets
