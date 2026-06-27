from __future__ import annotations

from typing import Optional

import pandas as pd


def find_repeated_prefix(
    sequence: str, min_prefix_len: int = 2
) -> Optional[tuple[str, int]]:
    """返回首个在后续位置再次出现的前缀及其重复起始位置。"""
    if not isinstance(sequence, str) or min_prefix_len < 1:
        return None

    for prefix_len in range(min_prefix_len, len(sequence)):
        prefix = sequence[:prefix_len]
        repeat_start = sequence.find(prefix, 1)
        if repeat_start != -1:
            return prefix, repeat_start
    return None


def find_repeated_prefix_psms(
    msms_df: pd.DataFrame,
    sequence_col: str = "Sequence",
    min_prefix_len: int = 2,
) -> pd.DataFrame:
    """筛出满足 `{pep}...{pep}` 模式的 PSM，并附加重复信息。"""
    if sequence_col not in msms_df.columns:
        raise ValueError(f"缺少列: {sequence_col}")

    matched_rows = []
    for _, row in msms_df.iterrows():
        sequence = row[sequence_col]
        if not isinstance(sequence, str):
            sequence = str(sequence)
        match = find_repeated_prefix(sequence, min_prefix_len=min_prefix_len)
        if match is None:
            continue
        prefix, repeat_start = match
        row_dict = row.to_dict()
        row_dict["Prefix"] = prefix
        row_dict["Repeat start"] = repeat_start
        matched_rows.append(row_dict)

    if not matched_rows:
        return pd.DataFrame(columns=[*msms_df.columns, "Prefix", "Repeat start"])
    return pd.DataFrame(matched_rows)


def select_top_repeated_prefix_psms(
    msms_df: pd.DataFrame,
    sequence_col: str = "Sequence",
    score_col: str = "Score",
    min_prefix_len: int = 3,
    top_n: int = 3,
) -> pd.DataFrame:
    """返回适合展示的高分 `{pep}...{pep}` PSM。"""
    matched = find_repeated_prefix_psms(
        msms_df, sequence_col=sequence_col, min_prefix_len=min_prefix_len
    )
    if matched.empty:
        return matched

    matched = matched.loc[matched["Repeat start"] >= matched["Prefix"].str.len()].copy()
    if matched.empty:
        return matched

    matched[score_col] = pd.to_numeric(matched[score_col], errors="coerce")
    matched = matched.sort_values(score_col, ascending=False)
    matched = matched.drop_duplicates(subset=[sequence_col], keep="first")
    return matched.head(top_n).reset_index(drop=True)
