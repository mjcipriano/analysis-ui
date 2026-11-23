"""Option schema helpers.

Every analysis and graph type declares its user-facing options with these
helpers. The frontend builds the whole options panel from the resulting
schema, so adding an option here is all that is needed to expose it in the UI.

Common keyword arguments accepted by every helper:

``section``  name of the collapsible panel section the option is shown in
``visible``  dict ``{other_option: value | [values]}``; the option is only
             shown (and only validated) when every condition matches
``help``     short hint shown under the control
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

CONTINUOUS = ("continuous",)
CATEGORICAL = ("nominal", "ordinal")
ANY = ("continuous", "nominal", "ordinal")


def _base(kind: str, name: str, label: str, default: Any, kw: dict) -> dict:
    opt = {"type": kind, "name": name, "label": label, "default": default}
    for key in ("section", "visible", "help"):
        if kw.get(key) is not None:
            opt[key] = kw.pop(key)
        else:
            kw.pop(key, None)
    if kw:
        raise TypeError(f"Unknown option arguments for {name!r}: {sorted(kw)}")
    return opt


def variable(name: str, label: str, permitted: Sequence[str] = CONTINUOUS,
             required: bool = True, **kw) -> dict:
    """A single column picker."""
    opt = _base("variable", name, label, None, kw)
    opt["permitted"] = list(permitted)
    opt["required"] = required
    return opt


def variables(name: str, label: str, permitted: Sequence[str] = CONTINUOUS,
              min: int = 1, max: int | None = None, **kw) -> dict:
    """A multi-column picker."""
    opt = _base("variables", name, label, [], kw)
    opt["permitted"] = list(permitted)
    opt["min"] = min
    opt["max"] = max
    return opt


def choice(name: str, label: str, choices: Iterable[tuple[str, str] | str],
           default: str | None = None, **kw) -> dict:
    """A dropdown. ``choices`` is a list of ``(value, label)`` tuples."""
    items = []
    for c in choices:
        if isinstance(c, str):
            items.append({"value": c, "label": c})
        else:
            items.append({"value": c[0], "label": c[1]})
    if default is None:
        default = items[0]["value"]
    opt = _base("choice", name, label, default, kw)
    opt["choices"] = items
    return opt


def boolean(name: str, label: str, default: bool = False, **kw) -> dict:
    return _base("bool", name, label, bool(default), kw)


def number(name: str, label: str, default: float | None = None,
           min: float | None = None, max: float | None = None,
           step: float | None = None, nullable: bool | None = None,
           placeholder: str | None = None, **kw) -> dict:
    """A numeric field. ``default=None`` means "auto" and makes it nullable."""
    opt = _base("number", name, label, default, kw)
    opt.update({"min": min, "max": max, "step": step,
                "nullable": default is None if nullable is None else nullable})
    if placeholder:
        opt["placeholder"] = placeholder
    return opt


def integer(name: str, label: str, default: int | None = None,
            min: int | None = None, max: int | None = None,
            nullable: bool | None = None, placeholder: str | None = None, **kw) -> dict:
    opt = number(name, label, default, min=min, max=max, step=1,
                 nullable=nullable, placeholder=placeholder, **kw)
    opt["type"] = "integer"
    return opt


def text(name: str, label: str, default: str = "", placeholder: str = "",
         multiline: bool = False, **kw) -> dict:
    opt = _base("text", name, label, default, kw)
    opt["placeholder"] = placeholder
    opt["multiline"] = multiline
    return opt


def level(name: str, label: str, sources: Sequence[str], **kw) -> dict:
    """Pick one group / factor level.

    ``sources`` lists option names. For a ``variable`` option the choices are
    that column's levels; for a ``variables`` option they are the selected
    column names. An empty value means "first".
    """
    opt = _base("level", name, label, "", kw)
    opt["sources"] = list(sources)
    return opt


def color(name: str, label: str, default: str = "", **kw) -> dict:
    """A colour picker. Empty string means "automatic"."""
    return _base("color", name, label, default, kw)


# --------------------------------------------------------------------------
# Reusable option groups
# --------------------------------------------------------------------------

def group_layout(paired: bool = False, section: str = "Data",
                 min_groups: int = 2, max_groups: int | None = None) -> list[dict]:
    """Options describing "groups of values", GraphPad *or* jamovi style.

    * ``layout = "wide"``  — every selected column is one group (Prism "column
      table"). For paired data each row is one subject.
    * ``layout = "long"``  — one column with values plus one grouping column
      (tidy data). For paired data an optional subject/ID column matches rows.

    Use :meth:`Context.groups`, :meth:`Context.long` or
    :meth:`Context.paired` to read the data in a layout independent way.
    """
    opts = [
        choice("layout", "Data layout", [
            ("wide", "Each column is a group"),
            ("long", "Values column + grouping column"),
        ], default="wide", section=section),
        variables("columns", "Groups (columns)", CONTINUOUS, min=min_groups,
                  max=max_groups, section=section, visible={"layout": "wide"}),
        variable("dep", "Values", CONTINUOUS, section=section,
                 visible={"layout": "long"}),
        variable("group", "Grouping variable", ANY, section=section,
                 visible={"layout": "long"}),
    ]
    if paired:
        opts.append(variable(
            "subject", "Subject / ID (pairs rows)", ANY, required=False,
            section=section, visible={"layout": "long"},
            help="Leave empty to pair by row order within each group."))
    return opts


P_ADJUST = [
    ("holm", "Holm"),
    ("bonferroni", "Bonferroni"),
    ("sidak", "Šidák"),
    ("fdr_bh", "FDR (Benjamini–Hochberg)"),
    ("none", "None"),
]


def alpha(section: str | None = None) -> dict:
    return number("alpha", "Significance level (α)", 0.05, min=0.0001, max=0.5,
                  step=0.01, section=section)


def defaults(schema: list[dict]) -> dict:
    return {o["name"]: o["default"] for o in schema}


def is_visible(opt: dict, values: dict) -> bool:
    cond = opt.get("visible")
    if not cond:
        return True
    for key, want in cond.items():
        have = values.get(key)
        if isinstance(want, (list, tuple)):
            if have not in want:
                return False
        elif have != want:
            return False
    return True
