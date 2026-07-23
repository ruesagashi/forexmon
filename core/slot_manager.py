"""
core/slot_manager.py
Manajemen slot posisi MT5 (50 slot total: 30 APPROVED, 15 PROBATION, 5 BUFFER).
Sesuai PRD Section 3.5.
"""

from loguru import logger
from typing import Dict, List, Optional
from collections import defaultdict

from execution.mt5_connector import connector
from core.pool_manager import pool_manager

MAX_TOTAL_SLOTS = 50
MAX_APPROVED_SLOTS = 30
MAX_PROBATION_SLOTS = 15
MAX_BUFFER_SLOTS = 5

class SlotManager:
    def __init__(self):
        pass

    def get_open_slots(self) -> Dict[str, int]:
        """
        Menghitung berapa slot yang sedang terpakai dan mengembalikan slot yang tersedia.
        Returns:
            Dict {"APPROVED": avail, "PROBATION": avail, "BUFFER": avail}
        """
        positions = connector.get_open_positions()
        
        used_approved = 0
        used_probation = 0
        used_unknown = 0
        
        for pos in positions:
            comment = pos["comment"]
            # Format comment: spec_{id}
            if comment and comment.startswith("spec_"):
                spec_id = comment.split("_")[1]
                spec = pool_manager.get_specialist_object(spec_id)
                if spec:
                    if spec.status == "APPROVED":
                        used_approved += 1
                    elif spec.status == "PROBATION":
                        used_probation += 1
                    else:
                        used_unknown += 1
                else:
                    used_unknown += 1
            else:
                used_unknown += 1
                
        avail_approved = max(0, MAX_APPROVED_SLOTS - used_approved)
        avail_probation = max(0, MAX_PROBATION_SLOTS - used_probation)
        
        total_used = used_approved + used_probation + used_unknown
        avail_buffer = max(0, MAX_TOTAL_SLOTS - total_used)
        
        return {
            "APPROVED": avail_approved,
            "PROBATION": avail_probation,
            "BUFFER": avail_buffer,
            "TOTAL_USED": total_used
        }

    def has_available_slot(self, specialist_status: str) -> bool:
        """
        Cek apakah masih ada slot kosong untuk status tertentu.
        Jika slot utama habis, bisa meminjam BUFFER jika tersedia.
        """
        slots = self.get_open_slots()
        
        if specialist_status == "APPROVED":
            return slots["APPROVED"] > 0 or slots["BUFFER"] > 0
        elif specialist_status == "PROBATION":
            return slots["PROBATION"] > 0 or slots["BUFFER"] > 0
            
        return False

slot_manager = SlotManager()
