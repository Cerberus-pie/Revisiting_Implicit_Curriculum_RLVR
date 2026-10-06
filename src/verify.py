"""Validate file hashes, CSV inventories, and every retained measurement grid."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def check_grid(records, fields, expected, label):
    keys = [tuple(float(row[k]) for k in fields) for row in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"Missing, unexpected, or duplicate measurement: {label}")


def main():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"File integrity check failed: {item['path']}")
    prov = json.loads((ROOT / "data/provenance.json").read_text())
    count = 0
    with zipfile.ZipFile(ROOT / "data/measurements.zip") as z:
        assert z.testzip() is None
        actual = [n for n in z.namelist() if n.endswith(".csv")]
        assert len(z.namelist()) == len(set(z.namelist()))
        assert set(actual) == {m["member"] for m in prov["measurements"]}
        for entry in prov["measurements"]:
            name = entry["member"]
            raw = z.read(name)
            assert hashlib.sha256(raw).hexdigest() == entry["sha256"], name
            records = list(csv.DictReader(io.StringIO(raw.decode())))
            assert len(records) == entry["rows"] and list(records[0]) == entry["columns"], name
            count += len(records)
            if name.startswith("runs/"):
                run = name.split("/")[1]
                config = json.loads((ROOT / f"configs/{run}.json").read_text())
                if name.endswith("eval.csv"):
                    check_grid(records, ("step", "length"),
                        {(s, l) for s in range(0, 40001, 50) for l in config["eval_lengths"]}, name)
                    for row in records:
                        assert int(row["n_eval"]) == 15360, name
                        for metric in ("greedy_success", "sampled_success", "greedy_trajectory_correct",
                                       "attention_mass", "attention_hit", "correct_transition_prob"):
                            assert 0 <= float(row[metric]) <= 1, (name, metric)
                else:
                    check_grid(records, ("step",), {(s,) for s in range(1, 40001)}, name)
                continue
            gradients = [r for r in records if r["kind"] == "gradient"]
            transfers = [r for r in records if r["kind"] == "transfer"]
            lengths = (5, 10, 15, 20, 40, 45) if ("5_10_20_40_45" in name or "extended" in name) else (5, 15, 45)
            interval = 2000 if "/transfer" in name else 100
            check_grid(gradients, ("step", "repeat", "length"),
                {(s, r, l) for s in range(0, 40001, interval) for r in range(3) for l in lengths}, name)
            for row in gradients:
                norm = float(row["reward_gradient_norm"])
                assert math.isfinite(norm) and norm >= 0, name
                values = json.loads(row["per_prompt_probabilities"])
                assert len(values) == 16 and all(0 <= v <= 1 for v in values), name
            if transfers:
                sources = (5, 10, 15, 20, 40) if len(lengths) == 6 else (5, 15)
                check_grid(transfers, ("step", "repeat", "source", "target", "epsilon"),
                    {(s, r, l, 45, e) for s in range(0, 40001, 2000) for r in range(3)
                     for l in sources for e in (.01, .005)}, name)
                for row in transfers:
                    assert abs(float(row["gradient_cosine"])) <= 1 + 1e-12, name
                    for field in ("positive_change", "negative_change", "target_probability"):
                        assert math.isfinite(float(row[field])), (name, field)
    print(json.dumps({"status": "passed", "files": len(manifest["files"]),
                      "measurement_tables": len(actual), "measurement_rows": count,
                      "report_version": 20}, indent=2))


if __name__ == "__main__":
    main()
