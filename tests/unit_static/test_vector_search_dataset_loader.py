from __future__ import annotations

import json
from pathlib import Path

from benchmarks.vector_search.dataset_loader import load_public_corpus


def test_load_public_corpus_from_local_beir_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "scifact"
    (dataset_root / "qrels").mkdir(parents=True)

    (dataset_root / "corpus.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "d1", "title": "One", "text": "alpha beta"}),
                json.dumps({"_id": "d2", "title": "Two", "text": "gamma delta"}),
                json.dumps({"_id": "d3", "title": "Three", "text": "epsilon zeta"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_root / "queries.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"_id": "q1", "text": "alpha topic"}),
                json.dumps({"_id": "q2", "text": "gamma topic"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n",
        encoding="utf-8",
    )

    dataset = load_public_corpus(
        "scifact",
        cache_dir=tmp_path,
        allow_download=False,
        max_docs=3,
        max_queries=1,
    )

    assert dataset.dataset_name == "beir_scifact_test"
    assert len(dataset.doc_ids) == 3
    assert len(dataset.queries) == 1
    assert dataset.queries[0].relevant_ids == ("d1",)
