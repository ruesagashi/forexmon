"""
config/symbols.py
Konfigurasi spesifik per symbol trading.
XAUUSD (Gold) memiliki karakteristik berbeda dari forex pairs biasa.
"""

from dataclasses import dataclass


@dataclass
class SymbolConfig:
    """Konfigurasi teknikal per symbol."""
    name: str
    description: str
    pip_size: float          # Ukuran 1 pip dalam harga
    lot_step: float          # Minimum kenaikan lot
    min_lot: float           # Minimum lot size
    max_lot: float           # Maximum lot size
    contract_size: float     # Ukuran kontrak (unit per lot)
    max_spread_pips: float   # Max spread yang bisa diterima (dalam pip)
    session_open: str        # Jam mulai sesi aktif (UTC)
    session_close: str       # Jam akhir sesi aktif (UTC)
    atr_period: int = 14     # Period ATR default
    atr_sl_mult: float = 1.5  # SL = N × ATR
    atr_tp_mult: float = 2.0  # TP = N × ATR
    min_volume_ratio: float = 0.8  # Min volume ratio untuk valid signal


# ─────────────────────────────────────────────────────────────────────────────
# XAUUSD — Gold vs USD
# Karakteristik:
# - 1 pip = 0.01 (karena harga dalam format 1234.56)
# - Contract size: 100 oz per lot
# - Sangat volatile terutama saat sesi London & New York overlap
# ─────────────────────────────────────────────────────────────────────────────
XAUUSD = SymbolConfig(
    name="XAUUSD",
    description="Gold vs US Dollar",
    pip_size=0.01,
    lot_step=0.01,
    min_lot=0.01,
    max_lot=50.0,
    contract_size=100.0,      # 100 oz per standard lot
    max_spread_pips=30.0,     # Spread max 30 pip (lebih toleran untuk Gold)
    session_open="07:00",     # UTC — London open
    session_close="21:00",    # UTC — NY close
    atr_sl_mult=1.5,
    atr_tp_mult=2.0,
    min_volume_ratio=0.9,     # Gold butuh volume konfirmasi lebih kuat
)

# ─────────────────────────────────────────────────────────────────────────────
# EURUSD — sebagai referensi (boleh ditambah nanti)
# ─────────────────────────────────────────────────────────────────────────────
EURUSD = SymbolConfig(
    name="EURUSD",
    description="Euro vs US Dollar",
    pip_size=0.0001,
    lot_step=0.01,
    min_lot=0.01,
    max_lot=100.0,
    contract_size=100000.0,
    max_spread_pips=3.0,
    session_open="07:00",
    session_close="21:00",
    atr_sl_mult=1.5,
    atr_tp_mult=2.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Registry — mapping nama symbol ke konfigurasinya
# ─────────────────────────────────────────────────────────────────────────────
SYMBOL_REGISTRY: dict[str, SymbolConfig] = {
    "XAUUSD": XAUUSD,
    "EURUSD": EURUSD,
}


def get_symbol_config(symbol: str) -> SymbolConfig:
    """
    Ambil konfigurasi untuk symbol tertentu.
    Raise ValueError jika symbol tidak dikenal.
    """
    if symbol not in SYMBOL_REGISTRY:
        raise ValueError(
            f"Symbol '{symbol}' tidak ditemukan di registry. "
            f"Tersedia: {list(SYMBOL_REGISTRY.keys())}"
        )
    return SYMBOL_REGISTRY[symbol]
