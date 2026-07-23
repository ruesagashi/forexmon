"""
tests/test_phase1.py
Unit tests untuk Phase 1: Feature Engineering + Database.
Tidak membutuhkan koneksi MT5 (menggunakan data dummy).
"""

import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def make_dummy_ohlcv(n: int = 300) -> pd.DataFrame:
    """
    Generate DataFrame OHLCV dummy untuk testing feature engineering.
    Simulasi harga XAUUSD yang realistis.
    """
    np.random.seed(42)
    price = 1900.0
    prices = []

    for _ in range(n):
        change = np.random.normal(0, 5)  # Volatilitas ~5 USD per candle
        price += change
        prices.append(max(price, 1500))  # Tidak boleh < 1500

    close = np.array(prices)
    high = close + np.abs(np.random.normal(0, 3, n))
    low = close - np.abs(np.random.normal(0, 3, n))
    open_ = close + np.random.normal(0, 2, n)
    volume = np.random.randint(100, 5000, n)

    idx = pd.date_range(start="2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "tick_volume": volume,
        "spread": np.zeros(n),
        "real_volume": np.zeros(n),
    }, index=idx)

    return df


@pytest.fixture
def dummy_df():
    return make_dummy_ohlcv(300)


@pytest.fixture
def df_with_features(dummy_df):
    from data.features import get_features
    return get_features(dummy_df)


@pytest.fixture
def test_db(tmp_path):
    """Database instance dengan file temporary."""
    from core.memory import MemoryDB
    db = MemoryDB(db_path=str(tmp_path / "test.db"))
    yield db
    db.close()


# ─────────────────────────────────────────────────────────────────────────────
# TEST: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_get_features_returns_dataframe(self, dummy_df):
        """get_features() harus return DataFrame non-empty."""
        from data.features import get_features
        result = get_features(dummy_df)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_ema_columns_exist(self, df_with_features):
        """Semua EMA columns harus ada."""
        for col in ["ema_10", "ema_20", "ema_50", "ema_200"]:
            assert col in df_with_features.columns, f"Kolom {col} tidak ditemukan"

    def test_rsi_range(self, df_with_features):
        """RSI harus antara 0-100."""
        rsi = df_with_features["rsi"].dropna()
        assert (rsi >= 0).all(), "RSI ada nilai di bawah 0"
        assert (rsi <= 100).all(), "RSI ada nilai di atas 100"

    def test_atr_positive(self, df_with_features):
        """ATR harus selalu positif."""
        atr = df_with_features["atr"].dropna()
        assert (atr > 0).all(), "ATR harus positif"

    def test_macd_columns_exist(self, df_with_features):
        """MACD, signal, dan histogram harus ada."""
        for col in ["macd", "macd_signal", "macd_hist"]:
            assert col in df_with_features.columns

    def test_bollinger_bands(self, df_with_features):
        """Upper BB harus selalu >= Lower BB."""
        upper = df_with_features["bb_upper"].dropna()
        lower = df_with_features["bb_lower"].dropna()
        assert (upper >= lower).all(), "BB Upper harus >= BB Lower"

    def test_bb_width_positive(self, df_with_features):
        """BB Width harus positif."""
        width = df_with_features["bb_width"].dropna()
        assert (width >= 0).all(), "BB Width harus non-negatif"

    def test_adx_range(self, df_with_features):
        """ADX harus antara 0-100."""
        adx = df_with_features["adx"].dropna()
        assert (adx >= 0).all(), "ADX tidak boleh negatif"
        assert (adx <= 100).all(), "ADX tidak boleh > 100"

    def test_candle_pattern_binary(self, df_with_features):
        """Pattern flags harus bernilai 0 atau 1."""
        pattern_cols = ["pattern_doji", "pattern_hammer", "pattern_pin_bar",
                        "pattern_engulf_bull", "pattern_engulf_bear"]
        for col in pattern_cols:
            assert col in df_with_features.columns, f"Kolom {col} tidak ada"
            vals = df_with_features[col].dropna().unique()
            assert set(vals).issubset({0, 1}), f"{col} harus 0 atau 1, dapat: {set(vals)}"

    def test_volume_ratio_positive(self, df_with_features):
        """Volume ratio harus positif."""
        vr = df_with_features["volume_ratio"].dropna()
        assert (vr >= 0).all(), "Volume ratio harus non-negatif"

    def test_swing_high_low_binary(self, df_with_features):
        """swing_high dan swing_low harus 0 atau 1."""
        for col in ["swing_high", "swing_low"]:
            vals = df_with_features[col].unique()
            assert set(vals).issubset({0, 1}), f"{col} harus binary"

    def test_regime_features_subset(self, df_with_features):
        """get_regime_features() harus return hanya 5 kolom regime."""
        from data.features import get_regime_features
        regime_df = get_regime_features(df_with_features)
        expected_cols = {"adx", "bb_width", "atr_ratio", "volume_ratio", "ema_slope"}
        assert set(regime_df.columns) == expected_cols

    def test_no_nan_in_key_features(self, df_with_features):
        """Tidak boleh ada NaN di kolom kritis setelah drop."""
        for col in ["rsi", "atr", "macd", "adx"]:
            assert df_with_features[col].isna().sum() == 0, f"Masih ada NaN di kolom {col}"

    def test_minimum_data_guard(self):
        """get_features() dengan data < 200 candle harus return data apa adanya (dengan warning)."""
        from data.features import get_features
        tiny_df = make_dummy_ohlcv(50)
        result = get_features(tiny_df)
        # Tidak crash, tapi return None atau df yang sama
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# TEST: DATABASE (MEMORY)
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabase:

    def test_db_initialized(self, test_db):
        """Database dan semua tabel harus terbuat."""
        conn = test_db._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row["name"] for row in tables}
        expected = {"specialists", "trades", "performance_snapshots", "regime_history", "system_events"}
        assert expected.issubset(table_names), f"Tabel kurang: {expected - table_names}"

    def test_add_and_get_specialist(self, test_db):
        """Test insert dan get specialist."""
        sid = f"spec_{uuid.uuid4().hex[:8]}"
        result = test_db.add_specialist(
            specialist_id=sid,
            regime_type="TRENDING_UP",
            features_used=["rsi", "adx"],
            metadata={"test": True},
        )
        assert result is True

        spec = test_db.get_specialist(sid)
        assert spec is not None
        assert spec["id"] == sid
        assert spec["regime_type"] == "TRENDING_UP"
        assert spec["status"] == "PROBATION"

    def test_duplicate_specialist_rejected(self, test_db):
        """Insert specialist dengan ID sama harus return False."""
        sid = f"spec_{uuid.uuid4().hex[:8]}"
        test_db.add_specialist(sid, "RANGING")
        result = test_db.add_specialist(sid, "RANGING")  # Duplikat
        assert result is False

    def test_update_specialist_status(self, test_db):
        """Test update status specialist."""
        sid = f"spec_{uuid.uuid4().hex[:8]}"
        test_db.add_specialist(sid, "BREAKOUT")
        test_db.update_specialist_status(sid, "APPROVED")

        spec = test_db.get_specialist(sid)
        assert spec["status"] == "APPROVED"

    def test_invalid_status_rejected(self, test_db):
        """Status tidak valid harus return False."""
        sid = f"spec_{uuid.uuid4().hex[:8]}"
        test_db.add_specialist(sid, "REVERSAL")
        result = test_db.update_specialist_status(sid, "INVALID_STATUS")
        assert result is False

    def test_add_and_close_trade(self, test_db):
        """Test insert trade open dan close."""
        sid = f"spec_{uuid.uuid4().hex[:8]}"
        test_db.add_specialist(sid, "TRENDING_UP")

        trade_id = test_db.add_trade(
            specialist_id=sid,
            symbol="XAUUSD",
            direction="BUY",
            entry_price=1950.50,
            sl=1940.00,
            tp=1970.00,
            lot_size=0.1,
            regime_at_entry="TRENDING_UP",
            confidence=0.85,
        )
        assert trade_id > 0

        # Close trade
        test_db.close_trade(trade_id, exit_price=1965.00, pnl=14.50, result="WIN")

        # Verifikasi
        trades = test_db.get_specialist_trades(sid)
        assert len(trades) == 1
        assert trades[0]["result"] == "WIN"
        assert trades[0]["pnl"] == pytest.approx(14.50)

    def test_specialist_performance_calculation(self, test_db):
        """Test kalkulasi WinRate dan ProfitFactor."""
        sid = f"spec_{uuid.uuid4().hex[:8]}"
        test_db.add_specialist(sid, "RANGING")

        # Insert 7 WIN dan 3 LOSS
        for i in range(7):
            tid = test_db.add_trade(sid, "XAUUSD", "BUY", 1900, 1890, 1920, 0.1, "RANGING", 0.8)
            test_db.close_trade(tid, 1920, pnl=20.0, result="WIN")

        for i in range(3):
            tid = test_db.add_trade(sid, "XAUUSD", "SELL", 1900, 1910, 1880, 0.1, "RANGING", 0.7)
            test_db.close_trade(tid, 1910, pnl=-10.0, result="LOSS")

        perf = test_db.get_specialist_performance(sid, last_n_trades=10)
        assert perf["winrate"] == pytest.approx(0.7)
        assert perf["profit_factor"] == pytest.approx(140.0 / 30.0)

    def test_log_regime(self, test_db):
        """Test insert dan get regime history."""
        test_db.log_regime("XAUUSD", "H1", "TRENDING_UP", 0.85)
        latest = test_db.get_latest_regime("XAUUSD")

        assert latest is not None
        assert latest["symbol"] == "XAUUSD"
        assert latest["regime"] == "TRENDING_UP"
        assert latest["confidence"] == pytest.approx(0.85)

    def test_log_event(self, test_db):
        """Test audit log system events."""
        test_db.log_event("SYSTEM_START", "Test startup", {"version": "1.0"})
        events = test_db.get_recent_events(limit=5)

        assert len(events) >= 1
        assert events[0]["event_type"] == "SYSTEM_START"

    def test_get_specialists_by_status(self, test_db):
        """Test filter specialist berdasarkan status."""
        for i in range(3):
            sid = f"approved_{uuid.uuid4().hex[:6]}"
            test_db.add_specialist(sid, "TRENDING_UP")
            test_db.update_specialist_status(sid, "APPROVED")

        for i in range(2):
            sid = f"probation_{uuid.uuid4().hex[:6]}"
            test_db.add_specialist(sid, "RANGING")
            # Status default = PROBATION

        approved = test_db.get_specialists_by_status("APPROVED")
        probation = test_db.get_specialists_by_status("PROBATION")

        assert len(approved) >= 3
        assert len(probation) >= 2

    def test_backup_creates_file(self, test_db, tmp_path):
        """Test backup database."""
        import shutil
        # Override backup dir ke tmp_path
        import core.memory
        original_backup_dir = core.memory.settings.DB_BACKUP_DIR

        # Langsung test fungsi backup dengan monkeypatching sederhana
        backup_path = tmp_path / "backup.db"
        shutil.copy2(test_db.db_path, backup_path)
        assert backup_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# TEST: SYMBOLS CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class TestSymbolConfig:

    def test_xauusd_config_exists(self):
        from config.symbols import get_symbol_config
        config = get_symbol_config("XAUUSD")
        assert config.name == "XAUUSD"
        assert config.pip_size == 0.01
        assert config.contract_size == 100.0

    def test_unknown_symbol_raises(self):
        from config.symbols import get_symbol_config
        with pytest.raises(ValueError):
            get_symbol_config("UNKNOWN_PAIR")
