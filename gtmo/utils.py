# gtmo/utils.py

"""utils.py
----------------------------------
Utility functions for the GTMØ package.

This module provides helper functions that can be used across
various components of the GTMØ library, such as type checking
for GTMØ specific entities.
"""

from __future__ import annotations
from typing import Any

# Importy względne dla komponentów z tego samego pakietu.
# Zakłada to, że plik core.py znajduje się w tym samym katalogu gtmo/.
try:
    from .core import O, AlienatedNumber
except ImportError:
    # Fallback na wypadek, gdyby utils.py był testowany lub używany w izolacji.
    # W prawidłowo zainstalowanym pakiecie import względny powinien zawsze działać.
    # To jest mniej prawdopodobne dla pliku utils, który zazwyczaj nie jest uruchamiany bezpośrednio.
    from core import O, AlienatedNumber


def is_ontological_singularity(item: Any) -> bool:
    """
    Sprawdza, czy podany element jest Ontologiczną Osobliwością (Ø).

    Args:
        item: Element do sprawdzenia.

    Returns:
        True jeśli element to Ontologiczna Osobliwość (Ø), w przeciwnym razie False.
    """
    return item is O  # O jest gwarantowanym singletonem z modułu core


def is_alienated_number(item: Any) -> bool:
    """
    Sprawdza, czy podany element jest Liczbą Wyalienowaną (ℓ∅).

    Args:
        item: Element do sprawdzenia.

    Returns:
        True jeśli element jest instancją AlienatedNumber, w przeciwnym razie False.
    """
    return isinstance(item, AlienatedNumber)


def is_gtmo_primitive(item: Any) -> bool:
    """
    Sprawdza, czy podany element jest podstawowym bytem prymitywnym GTMØ,
    tj. Ontologiczną Osobliwością (Ø) lub Liczbą Wyalienowaną (ℓ∅).

    Funkcja ta może być użyteczna do walidacji danych wejściowych lub
    kierowania logiką w zależności od tego, czy byt należy do
    podstawowego zestawu GTMØ.

    Args:
        item: Element do sprawdzenia.

    Returns:
        True jeśli element to Ø lub instancja AlienatedNumber, w przeciwnym razie False.
    """
    return is_ontological_singularity(item) or is_alienated_number(item)
