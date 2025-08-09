import redis
import json
import hashlib
from typing import Optional, Any
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class CacheManager:
    def __init__(self):
        self.enabled = settings.enable_cache
        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Running without cache.")
                self.enabled = False
                self.redis_client = None
        else:
            self.redis_client = None
    
    def _generate_key(self, query: str, prefix: str = "rag") -> str:
        """Generate cache key from query"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{prefix}:{query_hash}"
    
    def get(self, query: str) -> Optional[Any]:
        """Get cached result"""
        if not self.enabled or not self.redis_client:
            return None
        
        try:
            key = self._generate_key(query)
            cached = self.redis_client.get(key)
            if cached:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def set(self, query: str, value: Any, ttl: int = None):
        """Set cache value"""
        if not self.enabled or not self.redis_client:
            return
        
        try:
            key = self._generate_key(query)
            ttl = ttl or settings.cache_ttl
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(value, default=str)
            )
            logger.info(f"Cached result for query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def clear(self):
        """Clear all cache"""
        if self.enabled and self.redis_client:
            try:
                self.redis_client.flushdb()
                logger.info("Cache cleared")
            except Exception as e:
                logger.error(f"Cache clear error: {e}")