"""
EMP Config Vault v1.1

Configuration vault for the governance layer.
Manages secure configuration storage and access control.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigVault:
    """Secure configuration vault for governance layer."""
    
    def __init__(self, vault_path: Optional[str] = None):
        self.vault_path = Path(vault_path or "config/governance/vault.json")
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_cache: Dict[str, Any] = {}
        self._access_log: List[Dict[str, Any]] = []
        
        # Load existing configuration
        self._load_config()
        
        logger.info(f"Config Vault initialized with path: {self.vault_path}")
        
    def get_config(self, config_key: str, default: Any = None) -> Any:
        """Get configuration value."""
        try:
            value = self._config_cache.get(config_key, default)
            
            # Log access
            self._log_access('read', config_key, success=True)
            
            return value
            
        except Exception as e:
            self._log_access('read', config_key, success=False, error=str(e))
            logger.error(f"Error reading config {config_key}: {e}")
            return default
            
    def set_config(self, config_key: str, value: Any, description: Optional[str] = None):
        """Set configuration value."""
        try:
            old_value = self._config_cache.get(config_key)
            self._config_cache[config_key] = value
            
            # Add metadata
            if 'metadata' not in self._config_cache:
                self._config_cache['metadata'] = {}
            if 'configs' not in self._config_cache['metadata']:
                self._config_cache['metadata']['configs'] = {}
                
            self._config_cache['metadata']['configs'][config_key] = {
                'last_modified': datetime.now().isoformat(),
                'description': description or f"Configuration for {config_key}",
                'previous_value': old_value
            }
            
            # Save to file
            self._save_config()
            
            # Log access
            self._log_access('write', config_key, success=True, old_value=old_value, new_value=value)
            
            logger.info(f"Configuration updated: {config_key}")
            
        except Exception as e:
            self._log_access('write', config_key, success=False, error=str(e))
            logger.error(f"Error setting config {config_key}: {e}")
            
    def delete_config(self, config_key: str) -> bool:
        """Delete configuration value."""
        try:
            if config_key in self._config_cache:
                old_value = self._config_cache[config_key]
                del self._config_cache[config_key]
                
                # Remove metadata
                if 'metadata' in self._config_cache and 'configs' in self._config_cache['metadata']:
                    if config_key in self._config_cache['metadata']['configs']:
                        del self._config_cache['metadata']['configs'][config_key]
                        
                # Save to file
                self._save_config()
                
                # Log access
                self._log_access('delete', config_key, success=True, old_value=old_value)
                
                logger.info(f"Configuration deleted: {config_key}")
                return True
            else:
                logger.warning(f"Configuration not found: {config_key}")
                return False
                
        except Exception as e:
            self._log_access('delete', config_key, success=False, error=str(e))
            logger.error(f"Error deleting config {config_key}: {e}")
            return False
            
    def list_configs(self) -> List[str]:
        """List all configuration keys."""
        try:
            configs = [key for key in self._config_cache.keys() if key != 'metadata']
            
            # Log access
            self._log_access('list', 'all', success=True)
            
            return configs
            
        except Exception as e:
            self._log_access('list', 'all', success=False, error=str(e))
            logger.error(f"Error listing configs: {e}")
            return []
            
    def get_config_metadata(self, config_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a configuration."""
        try:
            if 'metadata' in self._config_cache and 'configs' in self._config_cache['metadata']:
                metadata = self._config_cache['metadata']['configs'].get(config_key)
                
                # Log access
                self._log_access('metadata', config_key, success=True)
                
                return metadata
            else:
                return None
                
        except Exception as e:
            self._log_access('metadata', config_key, success=False, error=str(e))
            logger.error(f"Error getting metadata for {config_key}: {e}")
            return None
            
    def export_config(self, export_path: str) -> bool:
        """Export configuration to external file."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'config': self._config_cache,
                'access_log': self._access_log[-100:]  # Last 100 access entries
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Configuration exported to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
            
    def import_config(self, import_path: str, overwrite: bool = False) -> bool:
        """Import configuration from external file."""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
                
            if 'config' not in import_data:
                logger.error("Invalid import file format")
                return False
                
            imported_config = import_data['config']
            
            # Check for conflicts
            conflicts = []
            for key in imported_config.keys():
                if key in self._config_cache and not overwrite:
                    conflicts.append(key)
                    
            if conflicts and not overwrite:
                logger.warning(f"Import conflicts found: {conflicts}. Use overwrite=True to resolve.")
                return False
                
            # Import configuration
            for key, value in imported_config.items():
                if key != 'metadata':  # Skip metadata during import
                    self._config_cache[key] = value
                    
            # Save to file
            self._save_config()
            
            logger.info(f"Configuration imported from: {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False
            
    def get_access_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get access log entries."""
        if limit:
            return self._access_log[-limit:]
        return self._access_log.copy()
        
    def clear_access_log(self):
        """Clear the access log."""
        self._access_log.clear()
        logger.info("Access log cleared")
        
    def _load_config(self):
        """Load configuration from file."""
        try:
            if self.vault_path.exists():
                with open(self.vault_path, 'r') as f:
                    self._config_cache = json.load(f)
                logger.info(f"Configuration loaded from {self.vault_path}")
            else:
                self._config_cache = {
                    'metadata': {
                        'created': datetime.now().isoformat(),
                        'version': '1.1.0',
                        'configs': {}
                    }
                }
                self._save_config()
                logger.info("New configuration vault created")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._config_cache = {}
            
    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(self.vault_path, 'w') as f:
                json.dump(self._config_cache, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            
    def _log_access(self, action: str, config_key: str, success: bool,
                   error: Optional[str] = None, old_value: Any = None, new_value: Any = None):
        """Log configuration access."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'config_key': config_key,
            'success': success,
            'error': error,
            'old_value': old_value,
            'new_value': new_value
        }
        
        self._access_log.append(log_entry)
        
        # Keep log size manageable
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-500:]
            
    def get_vault_statistics(self) -> Dict[str, Any]:
        """Get vault statistics."""
        config_count = len([key for key in self._config_cache.keys() if key != 'metadata'])
        access_count = len(self._access_log)
        
        # Calculate access statistics
        access_actions = {}
        for entry in self._access_log:
            action = entry.get('action', 'unknown')
            access_actions[action] = access_actions.get(action, 0) + 1
            
        return {
            'config_count': config_count,
            'access_count': access_count,
            'access_actions': access_actions,
            'vault_size_bytes': self.vault_path.stat().st_size if self.vault_path.exists() else 0,
            'last_modified': datetime.fromtimestamp(self.vault_path.stat().st_mtime).isoformat() if self.vault_path.exists() else None
        } 
