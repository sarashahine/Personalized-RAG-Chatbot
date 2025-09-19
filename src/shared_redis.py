import redis
import json
from typing import List, Dict

try:
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    r.ping()  # Test connection
    redis_available = True
except redis.ConnectionError:
    redis_available = False
    print("Redis not available, using in-memory fallback.")

class RedisHistoryManager:
    def __init__(self, client=None, max_messages: int = 40):
        if redis_available and client is None:
            self.r = r
        elif client:
            self.r = client
        else:
            self.r = None  # In-memory fallback
        self.max_messages = max_messages
        self.in_memory = {} if not redis_available else None

    def key(self, user_id: int) -> str:
        return f"history:{user_id}"

    def profile_key(self, user_id: int) -> str:
        return f"profile:{user_id}"

    def _ensure_list(self, key: str):
        if self.r:
            t = self.r.type(key)
            if t == 'none':
                return
            if t != 'list':
                self.r.delete(key)

    def add_message(self, user_id: int, role: str, content: str):
        if self.r:
            k = self.key(user_id)
            self._ensure_list(k)
            item = json.dumps({"role": role, "content": content})
            self.r.rpush(k, item)
            self.r.ltrim(k, -self.max_messages, -1)
        else:
            if user_id not in self.in_memory:
                self.in_memory[user_id] = {"history": [], "profile": {}}
            self.in_memory[user_id]["history"].append({"role": role, "content": content})
            self.in_memory[user_id]["history"] = self.in_memory[user_id]["history"][-self.max_messages:]

    def get_recent_history(self, user_id: int, max_messages: int = 20) -> List[Dict]:
        if self.r:
            k = self.key(user_id)
            self._ensure_list(k)
            raw = self.r.lrange(k, -max_messages, -1)
            return [json.loads(x) for x in raw]
        else:
            if user_id in self.in_memory:
                return self.in_memory[user_id]["history"][-max_messages:]
            return []

    def clear(self, user_id: int):
        if self.r:
            self.r.delete(self.key(user_id))
            self.r.delete(self.profile_key(user_id))
        else:
            if user_id in self.in_memory:
                self.in_memory[user_id] = {"history": [], "profile": {}}

    def set_user_profile(self, user_id: int, key: str, value: str):
        if self.r:
            pk = self.profile_key(user_id)
            profile = self.get_user_profile(user_id)
            profile[key] = value
            self.r.set(pk, json.dumps(profile))
        else:
            if user_id not in self.in_memory:
                self.in_memory[user_id] = {"history": [], "profile": {}}
            self.in_memory[user_id]["profile"][key] = value

    def get_user_profile(self, user_id: int) -> Dict:
        if self.r:
            pk = self.profile_key(user_id)
            data = self.r.get(pk)
            if data:
                return json.loads(data)
            return {}
        else:
            if user_id in self.in_memory:
                return self.in_memory[user_id]["profile"]
            return {}

    def extract_and_store_name(self, user_id: int, message: str):
        # Simple extraction for Arabic name introductions
        message_lower = message.lower()
        name = None
        if "ana esme" in message_lower or "اسمي" in message_lower:
            # Extract name after "ana esme" or "اسمي"
            if "ana esme" in message_lower:
                parts = message.split("ana esme")
            elif "اسمي" in message_lower:
                parts = message.split("اسمي")
            if len(parts) > 1:
                name = parts[1].strip().split()[0]  # Take first word
        elif "esme" in message_lower and len(message_lower.split()) > 1:
            # For "esme layth" or similar
            parts = message.split()
            if "esme" in parts:
                idx = parts.index("esme")
                if idx + 1 < len(parts):
                    name = parts[idx + 1]
        elif "ana" in message_lower and len(message_lower.split()) == 2:
            # For "ana layth"
            parts = message.split()
            if parts[0] == "ana" and len(parts) == 2:
                name = parts[1]
        if name:
            self.set_user_profile(user_id, "name", name)

def format_history_for_prompt(messages: List[Dict]) -> str:
    lines = []
    for m in messages:
        role = "user" if m.get("role") == "user" else "system"
        lines.append(f"{role}: {m.get('content','')}")
    return "\n".join(lines)


