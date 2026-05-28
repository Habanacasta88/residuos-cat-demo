"""Entry point para Streamlit Cloud.

Streamlit Cloud espera un archivo en la raíz del repo. Este archivo
delega al dashboard demo en src/residuos_cat/export/dashboard_demo.py.

Localmente puedes seguir usando:
    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Asegurar que src/ esté en path (Streamlit Cloud no instala el paquete)
REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from residuos_cat.export.dashboard_demo import main  # noqa: E402

if __name__ == "__main__":
    main()
