from typing import List, Dict, Optional
import time

# Global in-memory storage: {session_id: [{"role": "user", "content": "..."}, ...]}
# Limit to last 10 messages per session to prevent token overflow
MAX_HISTORY = 10 

class MemoryService:
    def __init__(self):
        self._storage: Dict[str, List[Dict]] = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self._storage:
            self._storage[session_id] = []
        
        self._storage[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Trim history
        if len(self._storage[session_id]) > MAX_HISTORY:
            self._storage[session_id] = self._storage[session_id][-MAX_HISTORY:]
            
    def get_history(self, session_id: str) -> List[Dict]:
        return self._storage.get(session_id, [])
    
    def clear_history(self, session_id: str):
        if session_id in self._storage:
            del self._storage[session_id]

# Singleton
_memory_service = MemoryService()

def get_memory_service() -> MemoryService:
    return _memory_service
