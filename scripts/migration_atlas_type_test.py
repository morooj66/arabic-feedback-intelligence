"""Phase M-MANUAL — generate a fully SYNTHETIC Extended JSON v2 type-test
package for Atlas import-mechanics verification only.

This script never connects to PostgreSQL and never connects to MongoDB.
Every document is hand-authored fake data -- no row from any real table
is read, transformed, or exported. It exists purely to let Atlas's
manual import be checked for BSON type fidelity ($date, null, bool,
int, string _id, UTF-8 Arabic text) without exposing any credential or
real visitor/business content.

Usage:
    .venv/bin/python scripts/migration_atlas_type_test.py --out-dir <path outside repo>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from rcs.migration.extjson import write_extended_json_array  # noqa: E402

# Fully synthetic -- not derived from, and not resembling, any real
# visitor comment, reviewer note, or business record. Arabic text here
# exists only to prove UTF-8 round-trips through Atlas's importer.
SYNTHETIC_DOCUMENTS = [
    {
        "_id": "atlas_type_test_doc_001",
        "synthetic_marker": True,
        "sample_string": "synthetic placeholder text",
        "sample_arabic_text": "هذا نص تجريبي وهمي لاختبار الاستيراد فقط، وليس تعليقاً حقيقياً.",
        "sample_date": dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
        "sample_null_field": None,
        "sample_boolean": True,
        "sample_integer": 42,
    },
    {
        "_id": "atlas_type_test_doc_002",
        "synthetic_marker": True,
        "sample_string": "another synthetic placeholder",
        "sample_arabic_text": "بيانات اختبارية غير حقيقية لفحص ترميز النص العربي.",
        "sample_date": dt.datetime(2026, 6, 15, 8, 30, 0, tzinfo=dt.timezone.utc),
        "sample_null_field": None,
        "sample_boolean": False,
        "sample_integer": 0,
    },
    {
        "_id": "atlas_type_test_doc_003",
        "synthetic_marker": True,
        "sample_string": "third synthetic placeholder",
        "sample_arabic_text": "مثال آخر لنص عربي وهمي بدون أي محتوى حقيقي.",
        "sample_date": dt.datetime(2025, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc),
        "sample_null_field": None,
        "sample_boolean": True,
        "sample_integer": -7,
    },
]


def _assert_outside_repo(out_dir: Path) -> None:
    real_out = out_dir.resolve()
    real_repo = ROOT.resolve()
    if real_out == real_repo or str(real_out).startswith(str(real_repo) + os.sep):
        raise RuntimeError(f"refusing to write export files inside the git repository: {real_out}")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Output directory -- MUST be outside the git repository.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    _assert_outside_repo(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "atlas_type_test.json"
    write_extended_json_array(SYNTHETIC_DOCUMENTS, path)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600

    manifest = {
        "filename": path.name,
        "document_count": len(SYNTHETIC_DOCUMENTS),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_of(path),
        "note": "fully synthetic -- no PostgreSQL or MongoDB connection made by this script",
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
