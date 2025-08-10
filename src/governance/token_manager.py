"""
cTrader Token Manager

Handles automated OAuth token refresh for cTrader API access.
Prevents the 30-day token expiry issue by automatically refreshing
access tokens before they expire.
"""

import asyncio
import logging
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import redis.asyncio as redis

try:
    from src.core.models import TokenData  # legacy
except Exception:  # pragma: no cover
    class TokenData:  # type: ignore
        pass
try:
    from src.core.configuration import SystemConfig  # legacy
except Exception:  # pragma: no cover
    class SystemConfig:  # type: ignore
        pass

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Manages cTrader OAuth tokens with automatic refresh capability.
    
    Features:
    - Automatic token refresh before expiry
    - Redis-based token storage
    - Configurable refresh thresholds
    - Health monitoring and alerts
    """
    
    def __init__(self, config: SystemConfig, redis_client: redis.Redis):
        """
        Initialize token manager.
        
        Args:
            config: System configuration with cTrader credentials
            redis_client: Redis client for token storage
        """
        self.config = config
        self.redis_client = redis_client
        self.token_key = "emp:ctrader_token"
        self.refresh_threshold_hours = 1  # Refresh 1 hour before expiry
        
        # cTrader OAuth endpoints
        self.token_url = "https://api.spotware.com/oauth2/token"
        self.client_id = config.trading.get('ctrader_client_id')
        self.client_secret = config.trading.get('ctrader_client_secret')
        self.refresh_token = config.trading.get('ctrader_refresh_token')
        
        logger.info("TokenManager initialized for cTrader API")
    
    async def get_valid_access_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.
        
        Returns:
            Valid access token string
            
        Raises:
            Exception: If token refresh fails
        """
        try:
            # Try to load existing token from Redis
            token_data = await self._load_token_from_redis()
            
            if token_data:
                # Check if token is still valid
                if not token_data.is_expired:
                    logger.debug("Using existing valid token")
                    return token_data.access_token
                
                # Token is expired or close to expiry, refresh it
                logger.info("Token expired or close to expiry, refreshing...")
                new_token = await self._refresh_access_token()
                return new_token.access_token
            
            # No token found, this is first time or Redis was cleared
            logger.info("No token found, performing initial token refresh")
            new_token = await self._refresh_access_token()
            return new_token.access_token
            
        except Exception as e:
            logger.error(f"Failed to get valid access token: {e}")
            raise
    
    async def _load_token_from_redis(self) -> Optional[TokenData]:
        """Load token data from Redis."""
        try:
            token_json = await self.redis_client.get(self.token_key)
            if token_json:
                token_dict = json.loads(token_json)
                return TokenData(**token_dict)
            return None
        except Exception as e:
            logger.warning(f"Failed to load token from Redis: {e}")
            return None
    
    async def _save_token_to_redis(self, token_data: TokenData) -> None:
        """Save token data to Redis."""
        try:
            token_dict = {
                'access_token': token_data.access_token,
                'token_type': token_data.token_type,
                'expires_in': token_data.expires_in,
                'refresh_token': token_data.refresh_token,
                'scope': token_data.scope,
                'created_at': token_data.created_at.isoformat()
            }
            
            # Store with expiry slightly longer than token lifetime
            expiry_seconds = token_data.expires_in + 3600  # 1 hour buffer
            await self.redis_client.setex(
                self.token_key,
                expiry_seconds,
                json.dumps(token_dict)
            )
            
            logger.info(f"Token saved to Redis, expires in {token_data.expires_in} seconds")
            
        except Exception as e:
            logger.error(f"Failed to save token to Redis: {e}")
            raise
    
    async def _refresh_access_token(self) -> TokenData:
        """
        Refresh the access token using the refresh token.
        
        Returns:
            New TokenData with refreshed access token
            
        Raises:
            Exception: If token refresh fails
        """
        if not self.refresh_token:
            raise ValueError("No refresh token configured")
        
        if not self.client_id or not self.client_secret:
            raise ValueError("cTrader client credentials not configured")
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    'grant_type': 'refresh_token',
                    'refresh_token': self.refresh_token,
                    'client_id': self.client_id,
                    'client_secret': self.client_secret
                }
                
                headers = {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
                
                async with session.post(
                    self.token_url,
                    data=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        token_response = await response.json()
                        
                        # Create new token data
                        new_token = TokenData(
                            access_token=token_response['access_token'],
                            token_type=token_response.get('token_type', 'Bearer'),
                            expires_in=token_response['expires_in'],
                            refresh_token=token_response.get('refresh_token', self.refresh_token),
                            scope=token_response.get('scope')
                        )
                        
                        # Save to Redis
                        await self._save_token_to_redis(new_token)
                        
                        logger.info(f"Token refreshed successfully, expires in {new_token.expires_in} seconds")
                        return new_token
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"Token refresh failed: {response.status} - {error_text}")
                        raise Exception(f"Token refresh failed: {response.status}")
                        
        except aiohttp.ClientError as e:
            logger.error(f"Network error during token refresh: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during token refresh: {e}")
            raise
    
    async def check_token_health(self) -> Dict[str, Any]:
        """
        Check the health status of the current token.
        
        Returns:
            Dictionary with token health information
        """
        try:
            token_data = await self._load_token_from_redis()
            
            if not token_data:
                return {
                    'healthy': False,
                    'message': 'No token found in Redis',
                    'expires_in': 0,
                    'needs_refresh': True
                }
            
            # Calculate time until expiry
            expires_at = token_data.created_at.timestamp() + token_data.expires_in
            now = datetime.now().timestamp()
            time_until_expiry = expires_at - now
            
            needs_refresh = time_until_expiry < (self.refresh_threshold_hours * 3600)
            
            return {
                'healthy': not token_data.is_expired,
                'message': 'Token is valid' if not token_data.is_expired else 'Token expired',
                'expires_in': int(time_until_expiry),
                'needs_refresh': needs_refresh,
                'refresh_threshold': self.refresh_threshold_hours * 3600
            }
            
        except Exception as e:
            logger.error(f"Error checking token health: {e}")
            return {
                'healthy': False,
                'message': f'Error: {str(e)}',
                'expires_in': 0,
                'needs_refresh': True
            }
    
    async def force_refresh(self) -> bool:
        """
        Force a token refresh regardless of current token state.
        
        Returns:
            True if refresh successful, False otherwise
        """
        try:
            new_token = await self._refresh_access_token()
            return new_token is not None
        except Exception as e:
            logger.error(f"Force refresh failed: {e}")
            return False
    
    async def start_token_monitor(self, check_interval: int = 3600) -> None:
        """
        Start background monitoring for token refresh.
        
        Args:
            check_interval: How often to check token health (seconds)
        """
        logger.info(f"Starting token monitor with {check_interval}s check interval")
        
        while True:
            try:
                health = await self.check_token_health()
                
                if health['needs_refresh']:
                    logger.info("Token needs refresh, initiating...")
                    await self._refresh_access_token()
                
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                logger.info("Token monitor stopped")
                break
            except Exception as e:
                logger.error(f"Error in token monitor: {e}")
                await asyncio.sleep(check_interval * 2)  # Back off on error
    
    def get_config_status(self) -> Dict[str, Any]:
        """
        Get configuration status for debugging.
        
        Returns:
            Dictionary with configuration status
        """
        return {
            'client_id_configured': bool(self.client_id),
            'client_secret_configured': bool(self.client_secret),
            'refresh_token_configured': bool(self.refresh_token),
            'redis_connected': self.redis_client is not None,
            'token_url': self.token_url
        }


# Example usage
async def main():
    """Example usage of TokenManager."""
    import os
    try:
        from src.core.configuration import load_config  # legacy
    except Exception:  # pragma: no cover
        def load_config(*args, **kwargs):  # type: ignore
            return {}
    
    # Load configuration
    config = load_config()
    
    # Create Redis client
    redis_client = redis.from_url(
        config.operational.get('redis', {}).get('url', 'redis://localhost:6379')
    )
    
    # Create token manager
    token_manager = TokenManager(config, redis_client)
    
    # Check configuration
    status = token_manager.get_config_status()
    print("Configuration status:", status)
    
    try:
        # Get valid token
        token = await token_manager.get_valid_access_token()
        print(f"Got valid token: {token[:20]}...")
        
        # Check health
        health = await token_manager.check_token_health()
        print("Token health:", health)
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
