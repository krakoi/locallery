"""Explicit real-model check on a small read-only library.

Run: uv run --extra cpu python scripts/smoke-model.py /path/to/photos
"""

import sys
from pathlib import Path

from locallery.config import Config
from locallery.service import Service

source = Path(sys.argv[1]).resolve()
storage = Path(".test-artifacts/python-real-model").resolve()
service = Service(
    Config(source, storage),
    lambda progress: (
        print(progress["stage"], progress["message"], flush=True)
        if not progress["busy"]
        else None
    ),
)
try:
    service.rescan()
    if service.progress["stage"] != "ready":
        raise RuntimeError(service.progress["message"])
    if service.progress["failed"]:
        raise RuntimeError(str(service.progress["errors"]))
    print("First scan:", service.progress)
    for query in ("a cat", "a bowl of soup"):
        result = service.search({"query": query})
        print(
            query,
            [(item["path"], round(item["score"], 3)) for item in result["items"][:3]],
        )
    items = service.get_images(1, 1)["items"]
    if items:
        reference = items[0]["id"]
        refined = service.search({"referenceImageId": reference, "query": "a cat"})
        assert all(item["id"] != reference for item in refined["items"])
        print("Refinement returned", refined["total"], "results")
    service.rescan()
    assert service.progress["stage"] == "ready" and service.progress["indexed"] == 0
    print("Unchanged scan reused", service.progress["unchanged"], "images")
finally:
    service.close()
