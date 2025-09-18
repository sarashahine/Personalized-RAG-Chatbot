import redis
import json
from typing import List, Dict

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

class RedisHistoryManager:
    def __init__(self, client=r, max_messages: int = 40):
        self.r = client
        self.max_messages = max_messages

    def key(self, user_id: int) -> str:
        return f"history:{user_id}"

    def profile_key(self, user_id: int) -> str:
        return f"profile:{user_id}"

    def _ensure_list(self, key: str):
        t = self.r.type(key)
        if t == 'none':
            return
        if t != 'list':
            self.r.delete(key)

    def add_message(self, user_id: int, role: str, content: str):
        k = self.key(user_id)
        self._ensure_list(k)
        item = json.dumps({"role": role, "content": content})
        self.r.rpush(k, item)
        self.r.ltrim(k, -self.max_messages, -1)

    def get_recent_history(self, user_id: int, max_messages: int = 20) -> List[Dict]:
        k = self.key(user_id)
        self._ensure_list(k)
        raw = self.r.lrange(k, -max_messages, -1)
        return [json.loads(x) for x in raw]

    def clear(self, user_id: int):
        self.r.delete(self.key(user_id))
        self.r.delete(self.profile_key(user_id))

    def set_user_profile(self, user_id: int, key: str, value: str):
        pk = self.profile_key(user_id)
        profile = self.get_user_profile(user_id)
        profile[key] = value
        self.r.set(pk, json.dumps(profile))

    def get_user_profile(self, user_id: int) -> Dict:
        pk = self.profile_key(user_id)
        data = self.r.get(pk)
        if data:
            return json.loads(data)
        return {}

    def extract_and_store_name(self, user_id: int, message: str):
        # Simple extraction for Arabic name introductions
        message_lower = message.lower()
        if "ana esme" in message_lower or "اسمي" in message_lower:
            # Extract name after "ana esme" or "اسمي"
            if "ana esme" in message_lower:
                parts = message.split("ana esme")
            elif "اسمي" in message_lower:
                parts = message.split("اسمي")
            if len(parts) > 1:
                name = parts[1].strip().split()[0]  # Take first word
                self.set_user_profile(user_id, "name", name)

def format_history_for_prompt(messages: List[Dict]) -> str:
    lines = []
    for m in messages:
        role = "user" if m.get("role") == "user" else "system"
        lines.append(f"{role}: {m.get('content','')}")
    return "\n".join(lines)


