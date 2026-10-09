#!/usr/bin/env python3
"""只读汇总 TAP-SID/GNPR 已完成结果，不训练、不分组、不输出原始用户信息。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import traceback
from collections import Counter, defaultdict, deque
from pathlib import Path


METRICS = ("R1", "R5", "R10", "NDCG10")
OFFICIAL_KEYS = ("recall@1", "recall@5", "recall@10", "ndcg@10")
SID_ATOM = re.compile(r"<[a-z]_\d+>")
SID_PATH = re.compile(r"(?:<[a-z]_\d+>\s*)+")
QUERY = re.compile(r"When\s+(.+?)\s+user_(\d+)\s+is likely to visit:", re.IGNORECASE)


def load_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def data_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"输入不存在：{path}")
    files = sorted(p for p in path.rglob("*") if p.is_file() and not p.name.startswith(("_", ".")))
    if not files:
        raise ValueError(f"输入目录无数据：{path}")
    return files


def json_records(path):
    # 与仓库 JSON 读取器相同，按分片文件名排序，跳过 Spark 标记文件。
    for part in data_files(path):
        with part.open(encoding="utf-8-sig") as handle:
            first = ""
            while not first:
                char = handle.read(1)
                if not char:
                    break
                first = char.strip()
            handle.seek(0)
            if first == "[":
                values = json.load(handle)
                if not isinstance(values, list):
                    raise ValueError("JSON 输入必须为对象列表")
                for row in values:
                    if not isinstance(row, dict):
                        raise ValueError("JSON 包含非对象记录")
                    yield row
            elif first:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError("JSONL 包含非对象记录")
                        yield row


def table_records(path, columns):
    for part in data_files(path):
        if part.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("读取已有 Parquet 需要 pyarrow：pip install pyarrow；不需要启动 Spark") from exc
            source = pq.ParquetFile(part)
            if not set(columns) <= set(source.schema_arrow.names):
                raise ValueError("Parquet 缺少所需字段")
            for batch in source.iter_batches(batch_size=65536, columns=list(columns)):
                yield from batch.to_pylist()
        elif part.suffix == ".csv":
            with part.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not set(columns) <= set(reader.fieldnames or []):
                    raise ValueError("CSV 缺少所需字段")
                yield from reader
        else:
            raise ValueError(f"不是支持的 CSV/Parquet 数据分片：{part}")


def one_existing(paths, label):
    found = [Path(p) for p in paths if Path(p).exists()]
    if len(found) != 1:
        raise ValueError(f"{label} 需要且只能找到一份输入；找到 {len(found)} 份，请移走旧副本或明确使用的运行目录")
    return found[0]


def sid(value):
    if isinstance(value, list):
        value = "".join(str(v) for v in value)
    return "".join(SID_ATOM.findall(str(value or "")))


def load_codebook(path):
    result, pids = {}, set()
    for row in table_records(path, ("pid", "sid_tokens")):
        pid, code = str(row["pid"]), sid(row["sid_tokens"])
        if not code or code in result or pid in pids:
            raise ValueError("码本含空 SID、重复 SID 或重复 PID")
        result[code] = pid
        pids.add(pid)
    if not result:
        raise ValueError("码本为空")
    return result


def normalized_query(text, codebook):
    match = QUERY.search(text)
    if not match:
        raise ValueError("无法解析测试查询的用户和目标时间")

    def replace_path(found):
        code = sid(found.group())
        if code not in codebook:
            raise ValueError("测试历史 SID 不在对应目录中")
        return "[PID:" + codebook[code] + "] "

    normalized = " ".join(SID_PATH.sub(replace_path, text).split())
    return match.group(2), match.group(1), normalized


def score(gold, ranked):
    if not isinstance(ranked, list):
        raise ValueError("predictions 必须是列表")
    # 与正式评估一致：规范化、按顺序去重、最多保留十个；允许不足十个。
    unique = list(dict.fromkeys(sid(value) for value in ranked if sid(value)))[:10]
    rank = unique.index(gold) + 1 if gold in unique else math.inf
    return [float(rank <= k) for k in (1, 5, 10)] + [1 / math.log2(rank + 1) if rank <= 10 else 0.0]


def load_method(root, name):
    root = Path(root)
    book_path = root / "codebook" / ("tap_sid.csv" if name == "tap" else "gnpr_sid.csv")
    dataset = one_existing([root / "data/llm_test.json", root / "data/llm_test.jsonl"], name + " 测试集")
    prediction_path, metric_path = root / "eval/test_predictions.json", root / "eval/test_metrics.json"
    codebook = load_codebook(book_path)
    data = list(json_records(dataset))
    predictions = list(json_records(prediction_path))
    official = load_json(metric_path)
    if not data or len(data) != len(predictions):
        raise ValueError(name + " 测试集与预测样本数不一致或为空")
    has_index = ["sample_index" in row for row in predictions]
    if any(has_index):
        indices = [row.get("sample_index") for row in predictions]
        if not all(type(i) is int for i in indices) or set(indices) != set(range(len(data))):
            raise ValueError(name + " sample_index 缺失、重复或覆盖不完整")
        predictions.sort(key=lambda row: row["sample_index"])
    samples, totals = [], [0.0] * 4
    for i, (row, pred) in enumerate(zip(data, predictions)):
        gold = sid(row.get("output", row.get("gold")))
        if gold not in codebook or sid(pred.get("gold")) != gold:
            raise ValueError(f"{name} 第 {i} 条预测 gold 与测试集不符")
        if "input" not in pred or pred["input"] != row.get("input"):
            raise ValueError(f"{name} 第 {i} 条预测 input 与测试集不符；请使用正式完整预测")
        ranked = pred.get("predictions")
        if not isinstance(ranked, list) or any(sid(v) not in codebook for v in ranked):
            raise ValueError(f"{name} 第 {i} 条预测含非法 SID 或无 predictions")
        values = score(gold, ranked)
        user, target_time, query = normalized_query(row["input"], codebook)
        samples.append({"key": (query, codebook[gold]), "user": user,
                        "time": target_time, "pid": codebook[gold], "scores": values})
        totals = [a + b for a, b in zip(totals, values)]
    reproduced = [v / len(data) for v in totals]
    for key, value in zip(OFFICIAL_KEYS, reproduced):
        expected = float(official[key])
        if not math.isfinite(expected) or not math.isclose(value, expected, abs_tol=1e-10, rel_tol=0):
            raise ValueError(name + " 复算指标与正式 metrics 不一致")
    for key in ("samples", "prediction_samples"):
        if key in official and int(official[key]) != len(data):
            raise ValueError(name + " 正式 metrics 样本数不一致")
    if "legal_sid_count" in official and int(official["legal_sid_count"]) != len(codebook):
        raise ValueError(name + " 正式 metrics 与实际候选目录数量不一致")
    if int(official.get("k", 10)) != 10:
        raise ValueError(name + " 本统计要求正式 top-10 评估结果")
    return codebook, samples, reproduced, [book_path, dataset, prediction_path, metric_path]


def metadata_cities(processed, path, catalog):
    mapping_path = one_existing([processed / "id_mappings.json", processed / "mappings/pois"], "POI 编号映射")
    if mapping_path.is_file():
        raw_mapping = load_json(mapping_path)["poi_id_to_internal"]
        pairs = ((str(k), str(v)) for k, v in raw_mapping.items())
    else:
        pairs = ((str(row["_poi"]), str(row["pid"])) for row in table_records(mapping_path, ("_poi", "pid")))
    mapping, internal = {}, set()
    for raw, pid in pairs:
        if raw in mapping or pid in internal:
            raise ValueError("原始 POI 映射不是一对一")
        mapping[raw] = pid
        internal.add(pid)
    if not catalog <= internal:
        raise ValueError("已有映射未覆盖实际候选目录")
    cities, seen, names = {}, set(), {}
    for row in json_records(path):
        raw = str(row["poi编号"])
        if raw in seen:
            raise ValueError("城市元数据含重复 POI 编号")
        seen.add(raw)
        pid = mapping.get(raw)
        if pid not in catalog:
            continue
        code, name = str(row.get("区县编码") or "").strip(), str(row.get("区县中文") or "").strip()
        if not code or not name:
            raise ValueError("实际候选 POI 缺少城市编码或名称")
        if code in names and names[code] != name:
            raise ValueError("同一城市编码对应多个名称")
        names[code] = name
        cities[pid] = (code, name)
    if set(cities) != catalog:
        raise ValueError(f"城市元数据未覆盖全部候选 POI，缺少 {len(catalog - set(cities))} 个；不得静默丢弃")
    return cities, [mapping_path, Path(path)], {"method": "explicit_metadata_via_existing_id_mapping",
        "metadata_pois": len(seen), "matched_catalog_pois": len(cities),
        "excluded_metadata_pois": len(seen) - len(cities)}


def point_in_ring(x, y, ring):
    # 点面匹配逻辑来自既有 industrial_city_r5_delivery；保留孔洞和边界点处理。
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-10 and min(x1, x2) - 1e-10 <= x <= max(x1, x2) + 1e-10 and min(y1, y2) - 1e-10 <= y <= max(y1, y2) + 1e-10:
            return True
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def geometry_contains(x, y, geometry):
    if geometry["type"] not in ("Polygon", "MultiPolygon"):
        raise ValueError("不支持的行政区几何类型")
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    return any(p and point_in_ring(x, y, p[0]) and not any(point_in_ring(x, y, hole) for hole in p[1:]) for p in polygons)


def boundary_cities(processed, catalog, log):
    root = Path(__file__).resolve().parents[1] / "industrial_city_r5_delivery/boundaries"
    province_path, city_dir = root / "province_full.json", root / "cities"
    direct = {110000, 120000, 310000, 500000, 810000, 820000}
    features = [f for f in load_json(province_path)["features"] if str(f["properties"].get("adcode", "")).isdigit() and int(f["properties"]["adcode"]) in direct]
    boundary_files = sorted(city_dir.glob("*_full.json"))
    if not boundary_files:
        raise ValueError("已有城市边界文件缺失")
    for path in boundary_files:
        features.extend(load_json(path)["features"])
    boundaries = []
    for feature in features:
        geom, props = feature["geometry"], feature["properties"]
        polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        points = [point for polygon in polygons for ring in polygon for point in ring]
        boundaries.append((str(props["adcode"]), str(props["name"]), geom,
                           (min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points))))
    source = one_existing([processed / "poi_info.csv", processed / "metadata/catalog"], "已有 POI 坐标")
    result, seen, unmatched, ambiguous = {}, set(), 0, 0
    started, last = time.monotonic(), time.monotonic()
    for row in table_records(source, ("pid", "longitude", "latitude")):
        pid = str(row["pid"])
        if pid in seen:
            raise ValueError("已有 POI 坐标含重复 PID")
        seen.add(pid)
        if pid not in catalog:
            continue
        x, y = float(row["longitude"]), float(row["latitude"])
        if not math.isfinite(x) or not math.isfinite(y) or not (-180 <= x <= 180 and -90 <= y <= 90):
            raise ValueError("POI 坐标无效")
        matches = [(code, name) for code, name, geom, (lo_x, lo_y, hi_x, hi_y) in boundaries
                   if lo_x <= x <= hi_x and lo_y <= y <= hi_y and geometry_contains(x, y, geom)]
        if not matches:
            result[pid] = ("UNMATCHED", "未匹配")
            unmatched += 1
        else:
            ambiguous += int(len(set(matches)) > 1)
            result[pid] = sorted(matches)[0]
        now = time.monotonic()
        if now - last >= 60:
            rate = len(result) / max(now - started, 0.001)
            log(f"城市匹配 {len(result)}/{len(catalog)}，{rate:.2f} POI/s，预计剩余 {(len(catalog)-len(result))/rate:.0f}s")
            last = now
    if set(result) != catalog:
        raise ValueError("已有坐标未覆盖实际候选目录")
    return result, [source, province_path, *boundary_files], {"method": "target_poi_coordinate_in_offline_boundaries",
        "unmatched_catalog_pois": unmatched, "ambiguous_catalog_pois": ambiguous,
        "ambiguous_rule": "choose smallest administrative code; retain count in validation",
        "coordinate_note": "使用预处理坐标；若非WGS84或边界附近归属存疑，优先使用显式城市元数据模式"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(path):
    digest, total, count = hashlib.sha256(), 0, 0
    path = Path(path)
    for part in data_files(path):
        digest.update((part.name if path.is_file() else part.relative_to(path).as_posix()).encode("utf-8"))
        with part.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
        count += 1
    return {"path": str(path.resolve()), "files": count, "bytes": total, "sha256_names_and_contents": digest.hexdigest()}


def analyze(tap_root, gnpr_root, processed_root, output_dir, poi_city_jsonl=None):
    output_dir, processed = Path(output_dir), Path(processed_root)
    output_dir.mkdir(parents=True, exist_ok=False)

    def log(message):
        with (output_dir / "analysis.log").open("a", encoding="utf-8") as handle:
            handle.write(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message + "\n")

    log("START 读取两种方法的正式测试集、预测和码本；仅CPU分析")
    tap_book, tap_rows, tap_overall, tap_paths = load_method(tap_root, "tap")
    gnpr_book, gnpr_rows, gnpr_overall, gnpr_paths = load_method(gnpr_root, "gnpr")
    catalog = set(tap_book.values())
    if catalog != set(gnpr_book.values()):
        raise ValueError("两种方法实际候选 POI 目录不一致")
    if len(tap_rows) != len(gnpr_rows):
        raise ValueError("两种方法测试样本数不一致")
    log("两种方法总体指标复算通过；开始按完整历史、用户、时间及目标对齐")
    buckets = defaultdict(deque)
    for row in gnpr_rows:
        buckets[row["key"]].append(row)
    aligned = []
    for row in tap_rows:
        if not buckets[row["key"]]:
            raise ValueError("两种方法测试样本无法按完整历史和目标 POI 对齐")
        aligned.append((row, buckets[row["key"]].popleft()))
    if any(buckets.values()):
        raise ValueError("GNPR 存在未对齐样本")
    log("样本对齐完成；读取城市归属")
    if poi_city_jsonl:
        cities, city_paths, city_validation = metadata_cities(processed, Path(poi_city_jsonl), catalog)
    else:
        cities, city_paths, city_validation = boundary_cities(processed, catalog, log)
    catalog_counts = Counter(cities.values())
    grouped = defaultdict(list)
    for tap, gnpr in aligned:
        grouped[cities[tap["pid"]]].append((tap, gnpr))

    def aggregate(pairs, code, name, count):
        n = len(pairs)
        record = {"city_code": code, "city": name, "catalog_pois": count,
                  "test_target_pois": len({a["pid"] for a, _ in pairs}),
                  "test_users": len({a["user"] for a, _ in pairs}), "test_samples": n,
                  "test_sample_share_pct": 100 * n / len(aligned)}
        for j, key in enumerate(METRICS):
            tap_value = sum(a["scores"][j] for a, _ in pairs) / n if n else None
            gnpr_value = sum(b["scores"][j] for _, b in pairs) / n if n else None
            delta = tap_value - gnpr_value if n else None
            record.update({f"tap_{key}": tap_value, f"gnpr_{key}": gnpr_value,
                           f"delta_{key}": delta, f"delta_{key}_pp": 100 * delta if n else None,
                           f"relative_{key}_pct": 100 * delta / gnpr_value if gnpr_value else None})
        return record

    records = [aggregate(grouped.get(key, []), key[0], key[1], count) for key, count in catalog_counts.items()]
    records.sort(key=lambda row: (-row["test_samples"], row["city_code"]))
    summary = aggregate(aligned, "ALL", "总体", len(catalog))
    summary.update({"catalog_cities": sum(row["city_code"] != "UNMATCHED" for row in records),
                    "test_cities": sum(row["test_samples"] > 0 and row["city_code"] != "UNMATCHED" for row in records),
                    "sum_city_test_users": sum(row["test_users"] for row in records),
                    "unmatched_test_samples": sum(row["test_samples"] for row in records if row["city_code"] == "UNMATCHED"),
                    "definitions": {
                        "city": "按测试目标POI所在城市归属，不是用户常住地；城市字段可能包含自治州/盟等行政地区",
                        "catalog_pois": "实际参与两种方法评估的共同候选目录，不是原始元数据文件总量",
                        "test_users": "本城市测试样本涉及的去重内部用户数；总体跨城去重，城市用户数不能相加当总体",
                        "metrics": "按测试样本计算，空测试城市指标为null，基线为0时相对提升为null",
                        "scope": "已有预测的分城市统计，不代表全部线上流量、A/B分组人数、人口规模或新的线上实验",
                        "grouping": "未划分大小规模，也未作显著性检验"}})
    if sum(row["test_samples"] for row in records) != summary["test_samples"] or sum(row["catalog_pois"] for row in records) != len(catalog):
        raise ValueError("城市计数与总体不一致")
    write_json(output_dir / "city_statistics.json", records)
    with (output_dir / "city_statistics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(output_dir / "summary.json", summary)
    columns = ["city", "catalog_pois", "test_users", "test_samples", "gnpr_R1", "tap_R1", "gnpr_R5", "tap_R5", "gnpr_R10", "tap_R10", "gnpr_NDCG10", "tap_NDCG10"]
    lines = ["# 城市级统计（未划分规模）", "", "统计口径详见 summary.json；下表不包含原始用户/POI编号或经纬度。", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in [summary, *records]:
        values = ["—" if row[key] is None else f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key]).replace("|", "\\|") for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "city_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("汇总表已写入文件；开始记录输入指纹")
    report = {"status": "PASS", "matched_samples": len(aligned), "catalog_pois": len(catalog),
              "query_and_target_alignment": "full normalized history + user + target time + target PID; multiset match",
              "duplicate_query_target_keys": sum(len(v) - 1 for v in _group_rows(tap_rows).values()),
              "city_mapping": city_validation,
              "aggregate_reproduced": {"tap": dict(zip(OFFICIAL_KEYS, tap_overall)), "gnpr": dict(zip(OFFICIAL_KEYS, gnpr_overall))},
              "inputs": [fingerprint(path) for path in [*tap_paths, *gnpr_paths, *city_paths]]}
    write_json(output_dir / "validation_report.json", report)
    log("DONE 校验通过，所有结果已写入文件")
    return report


def _group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["key"]].append(row)
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tap-run-root", type=Path, required=True)
    parser.add_argument("--gnpr-run-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--poi-city-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        print("输出目录已存在，拒绝覆盖；请使用新的输出目录。", file=sys.stderr)
        return 2
    try:
        analyze(args.tap_run_root, args.gnpr_run_root, args.processed_root, args.output_dir, args.poi_city_jsonl)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "validation_report.json", {"status": "FAILED", "error_type": type(exc).__name__, "message": str(exc)})
        with (args.output_dir / "analysis.log").open("a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        print(f"统计失败，详情见：{args.output_dir / 'analysis.log'}", file=sys.stderr)
        return 1
    print(f"统计完成，结果目录：{args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
