import time
import uuid
import base64
import secrets
from typing import Optional


def _int_to_base62(n: int) -> str:
    """Convert an integer to base62 encoding."""
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if n == 0:
        return chars[0]
    
    result = ""
    while n > 0:
        n, remainder = divmod(n, 62)
        result = chars[remainder] + result
    
    return result


def generate_id() -> str:
    """Generate a UUIDv7-compatible ID in base62 format."""
    # Create a UUIDv7-like value (timestamp + random)
    timestamp_ms = int(time.time() * 1000)
    timestamp_bytes = timestamp_ms.to_bytes(6, byteorder='big')
    random_bytes = secrets.token_bytes(10)  # 10 random bytes
    
    # Combine timestamp and random bytes
    combined_bytes = timestamp_bytes + random_bytes
    
    # Convert to integer and then to base62
    value = int.from_bytes(combined_bytes, byteorder='big')
    base62_id = _int_to_base62(value)
    
    # Ensure the base62 ID is always the expected length
    # Maximum theoretical length for 16 bytes in base62 is 22 chars
    padded_id = base62_id.rjust(22, '0')
    
    return padded_id


def new_id(prefix: str) -> str:
    """Create a new prefixed identifier."""
    return f"{prefix}-{generate_id()}"