"""
preparar_demo.py
----------------
Utilidad SOLO para la sustentación del Taller 2.

Antes de la demo, este script:
  1. Crea una copia de respaldo de la base (por si algo sale mal).
  2. Borra los últimos N artículos de 2026.

Así, al presentar, cuando le das al botón "Buscar artículos nuevos (2026)",
el scraping los vuelve a encontrar e insertar EN VIVO, demostrando que la
actualización automática funciona.

Uso:
    python preparar_demo.py            # borra 5 artículos de 2026 (por defecto)
    python preparar_demo.py 10         # borra 10
    python preparar_demo.py --restaurar  # restaura desde el respaldo
"""

import sys
import shutil
import sqlite3
from datetime import datetime

DB_PATH = "make_q1_2025.sqlite"
BACKUP_PATH = "make_q1_2025_BACKUP.sqlite"


def restaurar():
    try:
        shutil.copy(BACKUP_PATH, DB_PATH)
        print(f"✅ Base restaurada desde {BACKUP_PATH}")
    except FileNotFoundError:
        print(f"⚠️ No existe el respaldo {BACKUP_PATH}. No se puede restaurar.")


def preparar(n: int):
    # 1) Respaldo
    shutil.copy(DB_PATH, BACKUP_PATH)
    print(f"💾 Respaldo creado: {BACKUP_PATH}")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    total_antes = cur.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    n_2026 = cur.execute("SELECT COUNT(*) FROM papers WHERE year = 2026").fetchone()[0]
    print(f"Total actual: {total_antes} | de 2026: {n_2026}")

    if n_2026 == 0:
        print("⚠️ No hay artículos de 2026 para borrar. ¿Ya corriste el scraping?")
        con.close()
        return

    n = min(n, n_2026)

    # 2) Seleccionar los N artículos de 2026 con mayor paper_id (los últimos insertados)
    ids = [r[0] for r in cur.execute(
        "SELECT paper_id FROM papers WHERE year = 2026 ORDER BY paper_id DESC LIMIT ?",
        (n,),
    ).fetchall()]

    # Mostrar cuáles se van a borrar
    print(f"\nSe borrarán {len(ids)} artículo(s) de 2026 para la demo:")
    for r in cur.execute(
        f"SELECT doi, title FROM papers WHERE paper_id IN ({','.join('?'*len(ids))})",
        ids,
    ).fetchall():
        print(f"  - {r[0]} | {(r[1] or '')[:55]}")

    # 3) Borrar
    cur.execute(
        f"DELETE FROM papers WHERE paper_id IN ({','.join('?'*len(ids))})", ids
    )
    con.commit()

    total_despues = cur.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    con.close()

    print(f"\n✅ Listo. Total ahora: {total_despues} (se quitaron {total_antes - total_despues}).")
    print("Cuando presiones el botón en la app, el scraping los volverá a insertar.")
    print(f"Si quieres deshacer: python preparar_demo.py --restaurar")


if __name__ == "__main__":
    if "--restaurar" in sys.argv:
        restaurar()
    else:
        n = 5
        for a in sys.argv[1:]:
            if a.isdigit():
                n = int(a)
        preparar(n)
