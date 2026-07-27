#!/usr/bin/env python3
"""按测试样本数选择前八个城市，绘制 R@5 相对提升条形图。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CITY_ENGLISH = {
    "北京市": "Beijing",
    "天津市": "Tianjin",
    "上海市": "Shanghai",
    "重庆市": "Chongqing",
    "石家庄市": "Shijiazhuang",
    "太原市": "Taiyuan",
    "沈阳市": "Shenyang",
    "大连市": "Dalian",
    "长春市": "Changchun",
    "哈尔滨市": "Harbin",
    "南京市": "Nanjing",
    "苏州市": "Suzhou",
    "无锡市": "Wuxi",
    "杭州市": "Hangzhou",
    "宁波市": "Ningbo",
    "合肥市": "Hefei",
    "福州市": "Fuzhou",
    "厦门市": "Xiamen",
    "南昌市": "Nanchang",
    "济南市": "Jinan",
    "青岛市": "Qingdao",
    "郑州市": "Zhengzhou",
    "武汉市": "Wuhan",
    "长沙市": "Changsha",
    "广州市": "Guangzhou",
    "深圳市": "Shenzhen",
    "南宁市": "Nanning",
    "成都市": "Chengdu",
    "贵阳市": "Guiyang",
    "昆明市": "Kunming",
    "西安市": "Xi'an",
    "兰州市": "Lanzhou",
    "乌鲁木齐市": "Urumqi",
    "海口市": "Haikou",
    "香港特别行政区": "Hong Kong",
    "澳门特别行政区": "Macao",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-cities", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_top_cities(path: Path, limit: int) -> list[dict[str, object]]:
    if limit <= 0:
        raise ValueError("--top-cities 必须为正整数")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"城市指标文件不存在或为空：{path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"city", "samples", "rel_R5_pct"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"城市指标文件缺少字段：{sorted(missing)}")
        rows = [
            row
            for row in reader
            if row["city"] not in {"总体", "未匹配"}
        ]
    rows.sort(key=lambda row: (-int(row["samples"]), row["city"]))
    selected = rows[:limit]
    if len(selected) != limit:
        raise ValueError(
            f"可绘制城市数量不足：期望 {limit}，实际 {len(selected)}"
        )
    result: list[dict[str, object]] = []
    for row in selected:
        if not row["rel_R5_pct"].strip():
            raise ValueError(
                f"{row['city']} 的 Residual SID R@5 为 0，"
                "无法计算相对提升率"
            )
        result.append(
            {
                "city": CITY_ENGLISH.get(
                    row["city"],
                    row["city"].removesuffix("市"),
                ),
                "samples": int(row["samples"]),
                "relative_r5": float(row["rel_R5_pct"]),
            }
        )
    return result


def axis_limits(values: list[float]) -> tuple[float, float, float]:
    minimum = min(0.0, min(values))
    maximum = max(0.0, max(values))
    span = maximum - minimum
    if span <= 0:
        span = 1.0
    padding = max(0.35, span * 0.16)
    return minimum - (padding if minimum < 0 else 0), maximum + padding, padding


def draw_chart(
    rows: list[dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> None:
    # barh 从下向上排列，反转后使样本量最大的城市显示在最上方。
    plot_rows = list(reversed(rows))
    cities = [str(row["city"]) for row in plot_rows]
    values = [float(row["relative_r5"]) for row in plot_rows]
    x_min, x_max, padding = axis_limits(values)

    figure, axis = plt.subplots(
        figsize=(4.75, 4.85),
        facecolor="white",
    )
    bars = axis.barh(
        cities,
        values,
        height=0.62,
        color="#8FBC8F",
        edgecolor="#8FBC8F",
        linewidth=0.6,
        zorder=3,
    )

    axis.set_xlim(x_min, x_max)
    axis.set_xlabel("Top-5", fontsize=9.2)
    axis.axvline(0, color="#555555", linewidth=0.75, zorder=2)
    axis.xaxis.grid(
        True,
        color="#C8C8C8",
        linewidth=0.65,
        linestyle=(0, (2, 2)),
        zorder=0,
    )
    axis.yaxis.grid(False)

    label_offset = max(0.05, padding * 0.12)
    for bar, value in zip(bars, values, strict=True):
        if value >= 0:
            x = value + label_offset
            horizontal_alignment = "left"
        else:
            x = value - label_offset
            horizontal_alignment = "right"
        axis.text(
            x,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:+.1f}%",
            ha=horizontal_alignment,
            va="center",
            fontsize=8.4,
            fontweight="bold",
            color="#000000",
        )

    axis.tick_params(
        axis="y",
        labelsize=8.4,
        colors="#222222",
        length=3,
    )
    axis.tick_params(
        axis="x",
        labelsize=8.0,
        colors="#222222",
        length=3,
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    figure.subplots_adjust(
        left=0.22,
        right=0.97,
        top=0.97,
        bottom=0.12,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / (
        f"top{len(rows)}_city_r5_relative_improvement"
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            stem.with_suffix(f".{suffix}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    print(f"城市级 R@5 图已输出：{stem}")


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = load_top_cities(args.input, args.top_cities)
    draw_chart(rows, args.output_dir, args.dpi)
    for rank, row in enumerate(rows, start=1):
        print(
            f"rank={rank} city={row['city']} "
            f"samples={row['samples']} "
            f"relative_R5={row['relative_r5']:.6f}%"
        )


if __name__ == "__main__":
    main()
