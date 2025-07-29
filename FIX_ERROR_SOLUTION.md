# FIX Protocol Error Solution

## 🔍 Problem Analysis

**Error:** `'FIXApplication' object has no attribute 'set_message_queue'`  
**Location:** `main.py` lines 86-89  
**Root Cause:** The `FIXApplication` class doesn't have a `set_message_queue` method, but the main application is trying to call it.

## 🛠️ Solution

The issue is in `main.py` where it tries to call `set_message_queue()` on the FIX applications, but this method doesn't exist in the `FIXApplication` class.

### Option 1: Add the Missing Method (Recommended)

Add the `set_message_queue` method to the `FIXApplication` class:

**File:** `src/operational/fix_application.py`

Add this method to the `FIXApplication` class:

```python
def set_message_queue(self, queue):
    """
    Set the message queue for thread-safe communication.
    
    Args:
        queue: asyncio.Queue for message passing
    """
    self.external_queue = queue
    log.info(f"Message queue set for {self.connection_type} FIX application")
```

### Option 2: Modify Main Application Logic (Alternative)

**File:** `main.py`

Replace the problematic section (lines 86-89):

```python
# CURRENT (BROKEN):
if price_app:
    price_app.set_message_queue(price_queue)
if trade_app:
    trade_app.set_message_queue(trade_queue)

# REPLACE WITH:
if price_app:
    price_app.external_queue = price_queue
if trade_app:
    trade_app.external_queue = trade_queue
```

### Option 3: Remove Queue Assignment (Quick Fix)

**File:** `main.py`

Simply comment out or remove the problematic lines:

```python
# 3. Configure FIX applications with queues
price_app = self.fix_connection_manager.get_application("price")
trade_app = self.fix_connection_manager.get_application("trade")

# TODO: Implement proper queue integration
# if price_app:
#     price_app.set_message_queue(price_queue)
# if trade_app:
#     trade_app.set_message_queue(trade_queue)
```

## 🎯 Recommended Implementation

**I recommend Option 1** as it provides the most complete solution. Here's the exact code to add:

### Step 1: Update FIXApplication Class

Add this method to `src/operational/fix_application.py` after the existing methods:

```python
def set_message_queue(self, queue):
    """
    Set the external message queue for thread-safe communication.
    
    This allows the FIX application to forward messages to an external
    asyncio queue for processing by other components.
    
    Args:
        queue: asyncio.Queue for message passing between threads
    """
    self.external_queue = queue
    log.info(f"External message queue configured for {self.connection_type} FIX application")
    
def forward_to_external_queue(self, message):
    """
    Forward a message to the external queue if configured.
    
    Args:
        message: The FIX message to forward
    """
    if hasattr(self, 'external_queue') and self.external_queue:
        try:
            # Use asyncio.create_task to handle the async queue.put
            import asyncio
            asyncio.create_task(self.external_queue.put(message))
            log.debug(f"Message forwarded to external queue for {self.connection_type}")
        except Exception as e:
            log.error(f"Error forwarding message to external queue: {e}")
```

### Step 2: Update Message Processing

Modify the `on_message` method in `FIXApplication` to also forward to external queue:

```python
def on_message(self, message: simplefix.FixMessage):
    """
    Process incoming FIX messages.
    
    Args:
        message: The received FIX message
    """
    try:
        # ... existing message processing code ...
        
        # Forward to external queue if configured
        self.forward_to_external_queue(message)
        
    except Exception as e:
        log.error(f"Error processing message for {self.connection_type}: {e}")
```

## ✅ Verification

After implementing the fix, test with:

```bash
cd /home/ubuntu/emp_proving_ground_v1
python3 main.py --help
```

Expected result: Application should start without the `set_message_queue` error.

## 🔧 Alternative Quick Fix

If you want the fastest solution, simply add this one line to `FIXApplication`:

```python
def set_message_queue(self, queue):
    """Set message queue for external communication."""
    self.external_queue = queue
```

This minimal fix will resolve the immediate error and allow the application to start successfully.

