"""
execution/mt5_connector.py
Koneksi dan komunikasi dengan MetaTrader 5 terminal.
Handles: init, shutdown, reconnect, ambil data candle, info akun, posisi terbuka.
"""

import time
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

from config.settings import settings
from config.symbols import get_symbol_config

# Mapping timeframe string → MT5 constant
TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


class MT5Connector:
    """
    Wrapper untuk semua interaksi dengan MetaTrader 5.
    Gunakan sebagai context manager atau panggil initialize() secara manual.
    """

    def __init__(self, max_retries: int = 3, retry_delay: int = 5):
        self._connected: bool = False
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    # ─────────────────────────────────────────────────────────────────────────
    # Connection Management
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """
        Connect ke terminal MT5. Coba sampai max_retries kali.
        Return True jika berhasil, False jika gagal.
        """
        for attempt in range(1, self._max_retries + 1):
            logger.info(f"[MT5] Mencoba connect (attempt {attempt}/{self._max_retries})...")

            # Init dengan path terminal yang spesifik
            success = mt5.initialize(
                path=settings.MT5_PATH,
                login=settings.MT5_LOGIN if settings.MT5_LOGIN != 0 else None,
                password=settings.MT5_PASSWORD if settings.MT5_PASSWORD != "YOUR_PASSWORD" else None,
                server=settings.MT5_SERVER if settings.MT5_SERVER != "YOUR_BROKER_SERVER" else None,
            )

            if success:
                info = mt5.terminal_info()
                account = mt5.account_info()
                logger.success(
                    f"[MT5] Connected! "
                    f"Terminal: {info.name} | "
                    f"Build: {info.build} | "
                    f"Account: {account.login if account else 'N/A'} | "
                    f"Balance: {account.balance if account else 'N/A'}"
                )
                self._connected = True
                return True
            else:
                error = mt5.last_error()
                logger.warning(f"[MT5] Gagal connect (attempt {attempt}): {error}")
                if attempt < self._max_retries:
                    logger.info(f"[MT5] Retry dalam {self._retry_delay} detik...")
                    time.sleep(self._retry_delay)

        logger.error(f"[MT5] Gagal connect setelah {self._max_retries} attempts.")
        self._connected = False
        return False

    def shutdown(self) -> None:
        """Disconnect dari terminal MT5."""
        mt5.shutdown()
        self._connected = False
        logger.info("[MT5] Disconnected.")

    def is_connected(self) -> bool:
        """Cek apakah saat ini terhubung ke terminal MT5."""
        if not self._connected:
            return False
        # Double-check dengan ping ke MT5
        info = mt5.terminal_info()
        if info is None:
            self._connected = False
            logger.warning("[MT5] Koneksi terputus (terminal_info() returned None).")
        return self._connected

    def ensure_connected(self) -> bool:
        """
        Auto-reconnect jika koneksi terputus.
        Return True jika terkoneksi (atau berhasil reconnect).
        """
        if not self.is_connected():
            logger.warning("[MT5] Koneksi terputus, mencoba reconnect...")
            return self.initialize()
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Data Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        Ambil data candle OHLCV dari MT5.

        Args:
            symbol: Nama symbol (contoh: 'XAUUSD')
            timeframe: String timeframe (contoh: 'H1', 'M15')
            count: Jumlah candle yang diambil

        Returns:
            DataFrame dengan kolom: time, open, high, low, close, tick_volume, spread, real_volume
            None jika gagal.
        """
        if not self.ensure_connected():
            return None

        if timeframe not in TF_MAP:
            logger.error(f"[MT5] Timeframe tidak valid: {timeframe}. Pilihan: {list(TF_MAP.keys())}")
            return None

        # Pastikan symbol tersedia di terminal
        if not mt5.symbol_select(symbol, True):
            logger.error(f"[MT5] Symbol '{symbol}' tidak tersedia di terminal.")
            return None

        rates = mt5.copy_rates_from_pos(symbol, TF_MAP[timeframe], 0, count)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.error(f"[MT5] Gagal ambil candle {symbol} {timeframe}: {error}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]

        logger.debug(f"[MT5] {symbol} {timeframe}: {len(df)} candles diambil. Terakhir: {df.index[-1]}")
        return df

    def get_account_info(self) -> Optional[dict]:
        """
        Ambil informasi akun trading.

        Returns:
            Dict dengan: login, balance, equity, margin, free_margin,
            margin_level, profit, currency
        """
        if not self.ensure_connected():
            return None

        account = mt5.account_info()
        if account is None:
            logger.error(f"[MT5] Gagal ambil account info: {mt5.last_error()}")
            return None

        return {
            "login": account.login,
            "balance": account.balance,
            "equity": account.equity,
            "margin": account.margin,
            "free_margin": account.margin_free,
            "margin_level": account.margin_level,
            "profit": account.profit,
            "currency": account.currency,
        }

    def get_open_positions(self, symbol: Optional[str] = None) -> list[dict]:
        """
        Ambil semua posisi terbuka, opsional filter per symbol.

        Returns:
            List of dict: ticket, symbol, type, volume, price_open,
            sl, tp, profit, comment, magic, time
        """
        if not self.ensure_connected():
            return []

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "comment": pos.comment,
                "magic": pos.magic,
                "time": datetime.fromtimestamp(pos.time),
            })

        return result

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Ambil info teknikal symbol dari MT5 (spread, point, digits, dll).
        """
        if not self.ensure_connected():
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"[MT5] Symbol info untuk '{symbol}' tidak ditemukan.")
            return None

        return {
            "name": info.name,
            "bid": info.bid,
            "ask": info.ask,
            "spread": info.spread,
            "digits": info.digits,
            "point": info.point,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
        }

    def get_server_time(self) -> Optional[datetime]:
        """Ambil waktu server MT5."""
        if not self.ensure_connected():
            return None
        tick = mt5.symbol_info_tick("XAUUSD")
        if tick:
            return datetime.fromtimestamp(tick.time)
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Order Execution
    # ─────────────────────────────────────────────────────────────────────────
    
    def _get_filling_mode(self, symbol: str) -> int:
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        
        # Cek supported filling modes dari broker
        # 1 = FOK, 2 = IOC
        filling = info.filling_mode
        if filling & 1:
            return mt5.ORDER_FILLING_FOK
        elif filling & 2:
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN

    def send_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment: str = ""
    ) -> Optional[dict]:
        """
        Kirim market order BUY atau SELL.
        """
        if not self.ensure_connected():
            return None

        mt5_type = mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": 1001,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol),
        }

        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"[MT5] Gagal send_order: {mt5.last_error()}")
            return None
            
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[MT5] Order ditolak: retcode={result.retcode}, comment={result.comment}")
            return None
            
        logger.success(f"[MT5] Order berhasil: {order_type} {volume} lot {symbol}. Ticket: {result.order}")
        return {
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume
        }

    def close_position(self, ticket: int, symbol: str, position_type: str, volume: float) -> bool:
        """Tutup sebuah posisi terbuka."""
        if not self.ensure_connected():
            return False

        # Opposite type for closing
        mt5_type = mt5.ORDER_TYPE_SELL if position_type.upper() == "BUY" else mt5.ORDER_TYPE_BUY
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return False
            
        price = tick.bid if mt5_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5_type,
            "position": ticket,
            "price": float(price),
            "deviation": 20,
            "magic": 1001,
            "comment": "Close pos",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol),
        }

        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"[MT5] Gagal close posisi {ticket}: {mt5.last_error()}")
            return False
            
        logger.success(f"[MT5] Posisi {ticket} ditutup pada harga {result.price}")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Context Manager Support
    # ─────────────────────────────────────────────────────────────────────────

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


# Singleton instance
connector = MT5Connector()
