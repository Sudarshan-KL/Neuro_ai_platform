from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.disorder_models import DisorderModelService


def main() -> None:
    service = DisorderModelService()
    service.ensure_artifacts()
    info = service.model_info()
    print("Disorder models trained/available:")
    for name, meta in info.items():
        print(f"- {name}: {meta['artifact']}")


if __name__ == "__main__":
    main()

