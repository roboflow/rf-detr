# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""MkDocs hook for exposing cookbook card data to documentation templates."""

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]  # yaml stubs not in docs group


def _load_cards(cards_path: Path) -> list[dict[str, Any]]:
    """Load cookbook card definitions from a YAML file.

    Reads the YAML file and returns the ``cards`` list, which is consumed by the
    cookbook landing-page template. Centralising card data in YAML keeps content
    decoupled from presentation and avoids repeated edits to the HTML template.

    Args:
        cards_path: Path to the cookbook cards YAML file.

    Returns:
        Ordered list of card dictionaries from the YAML ``cards`` key.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        RuntimeError: If the YAML payload does not expose a list under ``cards``.
        yaml.YAMLError: If the YAML file cannot be parsed.

    Example:
        >>> cards = _load_cards(Path("docs/cookbooks/cards.yaml"))
        >>> isinstance(cards, list)
        True
    """
    with cards_path.open("r", encoding="utf-8") as cards_file:
        payload = yaml.safe_load(cards_file)
    cards = (payload or {}).get("cards")
    if not isinstance(cards, list):
        msg = f"Missing list 'cards' in {cards_path}"
        raise RuntimeError(msg)
    return cards


def on_config(config: dict[str, Any]) -> dict[str, Any]:
    """Expose cookbook card data to MkDocs templates.

    Adds ``config.extra.cookbooks_cards`` so the cookbooks landing-page template
    can render cards from a single YAML source of truth instead of hardcoded
    HTML blocks.

    Args:
        config: MkDocs configuration object.

    Returns:
        Updated MkDocs configuration object.

    Raises:
        FileNotFoundError: If the cookbook cards YAML file is missing.
        RuntimeError: If the YAML file does not expose a ``cards`` list.

    Example:
        >>> cfg = {"extra": {}}
        >>> updated = on_config(cfg)
        >>> isinstance(updated["extra"]["cookbooks_cards"], list)
        True
    """
    cards_path = Path(__file__).resolve().parents[2] / "docs" / "cookbooks" / "cards.yaml"
    extra = config.setdefault("extra", {})
    extra["cookbooks_cards"] = _load_cards(cards_path)
    return config
