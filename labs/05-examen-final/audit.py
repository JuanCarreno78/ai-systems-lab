"""Log de auditoría en JSONL: una línea por evento de seguridad.

Guarda fecha, telegram_user_id, evento, detalle y, en los logins, la similitud obtenida.
Nunca guarda imágenes, videos ni embeddings.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import AUDITORIA_FILE


def registrar(evento: str, telegram_user_id: int | None, archivo: Path = AUDITORIA_FILE, **detalle: Any) -> None:
    linea = {"fecha": datetime.now().isoformat(timespec="seconds"), "telegram_user_id": telegram_user_id,
             "evento": evento, **detalle}
    with archivo.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(linea, ensure_ascii=False) + "\n")
