"""Merge keys into a JSON column without overwriting keys other writers set concurrently."""

from __future__ import annotations

import json

from django.db import connection, transaction


def merge_json_field(manager, pk, field: str, patch: dict, *, remove_keys: tuple[str, ...] = ()) -> int:
    """
    UPDATE ... SET field = (field - remove_keys) || patch  for one row.

    Postgres does it in one statement (jsonb operators). Other databases (tests) lock the row, merge in
    Python and save. Returns the number of rows updated.
    """
    patch = dict(patch or {})
    if not patch and not remove_keys:
        return 0
    if connection.vendor == "postgresql":
        from django.db.models.expressions import RawSQL

        column = manager.model._meta.get_field(field).column
        expr = f"COALESCE({connection.ops.quote_name(column)}, '{{}}'::jsonb)"
        params: list = []
        for key in remove_keys:
            expr = f"({expr} - %s)"
            params.append(key)
        expr = f"{expr} || %s::jsonb"
        params.append(json.dumps(patch, default=str))
        return manager.filter(pk=pk).update(**{field: RawSQL(expr, params)})

    with transaction.atomic():
        row = manager.select_for_update().filter(pk=pk).only("pk", field).first()
        if row is None:
            return 0
        current = getattr(row, field)
        data = dict(current) if isinstance(current, dict) else {}
        for key in remove_keys:
            data.pop(key, None)
        data.update(json.loads(json.dumps(patch, default=str)))
        return manager.filter(pk=pk).update(**{field: data})
