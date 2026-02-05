from __future__ import annotations
from typing import List, Dict

def steps_to_text(steps: List[Dict], with_role_header: bool = True) -> str:
    rows = []
    for i, s in enumerate(steps, start=1):
        action = (s.get("action") or s.get("description") or "").strip()
        role = (s.get("role") or "").strip()
        rows.append((str(i), action, role))

    if not rows:
        return "Шаг | Роль\n"

    w_step = max([len(r[0]) for r in rows] + [3])
    w_act = max([len(r[1]) for r in rows] + [3])
    w_role = max([len(r[2]) for r in rows] + [4])

    lines = []
    if with_role_header:
        lines.append(f"{'Шаг'.ljust(w_step)} | {'Роль'.ljust(w_role)}")
    for step, action, role in rows:
        if with_role_header:
            lines.append(f"{(step+'.').ljust(w_step)} {action.ljust(w_act)} | {role.ljust(w_role)}")
        else:
            lines.append(f"{step}. {action}")
    return "\n".join(lines).strip() + "\n"
