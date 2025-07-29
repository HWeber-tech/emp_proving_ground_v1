"""
Real-Time Data Streaming System for EMP

This module implements comprehensive real-time data streaming capabilities
including WebSocket management, stream processing, event handling, and
connection management with robust error recovery.

Author: EMP Development Team
Date: July 18, 2024
Phase: 2 - Advanced Data Integration
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import websockets
import aiohttp
from asyncio import Queue, Semaphore
import numpy as np

from src.sensory.core.base import MarketData, InstrumentMeta

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of data streams"""
    MARKET_DATA = "market_data"
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    NEWS = "news"
    ECONOMIC = "economic"
    SENTIMENT = "sentiment"


class ConnectionStatus(Enum):
    """WebSocket connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Configuration for data streams"""
    stream_type: StreamType
    url: str
    symbols: List[str]
    heartbeat_interval: int = 30
    reconnect_delay: int = 5
    max_reconnect_attempts: int = 10
    connection_timeout: int = 30
    message_timeout: int = 60
    buffer_size: int = 1000
    batch_size: int = 100
    enable_compression: bool = True


@dataclass
class StreamEvent:
    """Event from data stream"""
    stream_type: StreamType
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    sequence_number: int
    is_heartbeat: bool = False
    is_error: bool = False


@dataclass
class StreamMetrics:
    """Metrics for stream performance"""
    messages_received: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    last_message_time: Optional[datetime] = None
    average_latency_ms: float = 0.0
    uptime_seconds: float = 0.0
    error_count: int = 0


class WebSocketManager:
    """
    Manages WebSocket connections with robust error handling and reconnection logic.
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.websocket = None
        self.status = ConnectionStatus.DISCONNECTED
        self.reconnect_attempts = 0
        self.last_heartbeat = None
        self.connection_start_time = None
        self.metrics = StreamMetrics()
        
        # Event queues
        self.message_queue = Queue(maxsize=config.buffer_size)
        self.control_queue = Queue(maxsize=100)
        
        # Semaphores for connection management
        self.connection_semaphore = Semaphore(1)
        self.processing_semaphore = Semaphore(10)
        
        # Event handlers
        self.on_message: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_connect: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None
        
        # Background tasks
        self._connection_task = None
        self._heartbeat_task = None
        self._processing_task = None
        
        logger.info(f"WebSocketManager initialized for {config.stream_type.value}")
    
    async def connect(self) -> bool:
        """Connect to WebSocket with retry logic"""
        async with self.connection_semaphore:
            if self.status in [ConnectionStatus.CONNECTING, ConnectionStatus.CONNECTED]:
                return True
            
            self.status = ConnectionStatus.CONNECTING
            self.metrics.connection_attempts += 1
            
            try:
                logger.info(f"Connecting to {self.config.url}")
                
                # Connect with timeout
                self.websocket = await asyncio.wait_for(
                    websockets.connect(
                        self.config.url,
                        compression=self.config.enable_compression,
                        ping_interval=20,
                        ping_timeout=10
                    ),
                    timeout=self.config.connection_timeout
                )
                
                self.status = ConnectionStatus.CONNECTED
                self.reconnect_attempts = 0
                self.connection_start_time = datetime.utcnow()
                self.metrics.successful_connections += 1
                
                # Start background tasks
                self._start_background_tasks()
                
                # Subscribe to symbols
                await self._subscribe_to_symbols()
                
                if self.on_connect:
                    await self.on_connect()
                
                logger.info(f"Successfully connected to {self.config.url}")
                return True
                
            except Exception as e:
                self.status = ConnectionStatus.ERROR
                self.metrics.failed_connections += 1
                self.metrics.error_count += 1
                logger.error(f"Failed to connect to {self.config.url}: {e}")
                
                if self.on_error:
                    await self.on_error(f"Connection failed: {e}")
                
                return False
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket"""
        self.status = ConnectionStatus.DISCONNECTED
        
        # Stop background tasks
        self._stop_background_tasks()
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
        
        if self.on_disconnect:
            await self.on_disconnect()
        
        logger.info("WebSocket disconnected")
    
    async def _start_background_tasks(self) -> None:
        """Start background processing tasks"""
        self._connection_task = asyncio.create_task(self._connection_monitor())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        self._processing_task = asyncio.create_task(self._message_processor())
    
    def _stop_background_tasks(self) -> None:
        """Stop background processing tasks"""
        if self._connection_task:
            self._connection_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._processing_task:
            self._processing_task.cancel()
    
    async def _connection_monitor(self) -> None:
        """Monitor connection and handle reconnection"""
        while self.status == ConnectionStatus.CONNECTED:
            try:
                # Listen for messages
                async for message in self.websocket:
                    await self.message_queue.put(message)
                    self.metrics.messages_received += 1
                    self.metrics.last_message_time = datetime.utcnow()
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.metrics.error_count += 1
                break
        
        # Attempt reconnection if not manually disconnected
        if self.status != ConnectionStatus.DISCONNECTED:
            await self._attempt_reconnection()
    
    async def _attempt_reconnection(self) -> None:
        """Attempt to reconnect with exponential backoff"""
        while (self.status != ConnectionStatus.DISCONNECTED and 
               self.reconnect_attempts < self.config.max_reconnect_attempts):
            
            self.status = ConnectionStatus.RECONNECTING
            self.reconnect_attempts += 1
            
            # Exponential backoff
            delay = min(self.config.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 300)
            logger.info(f"Attempting reconnection {self.reconnect_attempts}/{self.config.max_reconnect_attempts} in {delay}s")
            
            await asyncio.sleep(delay)
            
            if await self.connect():
                return
        
        # Max reconnection attempts reached
        self.status = ConnectionStatus.ERROR
        logger.error("Max reconnection attempts reached")
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor heartbeat and connection health"""
        while self.status == ConnectionStatus.CONNECTED:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                # Check if we've received messages recently
                if (self.metrics.last_message_time and 
                    (datetime.utcnow() - self.metrics.last_message_time).total_seconds() > self.config.message_timeout):
                    logger.warning("No messages received, connection may be stale")
                    break
                
                # Send heartbeat if required
                if self.websocket:
                    try:
                        await self.websocket.ping()
                        self.last_heartbeat = datetime.utcnow()
                    except Exception as e:
                        logger.error(f"Heartbeat failed: {e}")
                        break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                break
    
    async def _message_processor(self) -> None:
        """Process messages from the queue"""
        while self.status in [ConnectionStatus.CONNECTED, ConnectionStatus.RECONNECTING]:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                async with self.processing_semaphore:
                    await self._process_message(message)
                    self.metrics.messages_processed += 1
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                self.metrics.error_count += 1
                self.metrics.messages_dropped += 1
    
    async def _process_message(self, message: str) -> None:
        """Process individual message"""
        try:
            data = json.loads(message)
            
            # Create stream event
            event = StreamEvent(
                stream_type=self.config.stream_type,
                symbol=data.get('symbol', 'UNKNOWN'),
                timestamp=datetime.utcnow(),
                data=data,
                source=self.config.url,
                sequence_number=self.metrics.messages_processed
            )
            
            # Call message handler
            if self.on_message:
                await self.on_message(event)
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
            self.metrics.error_count += 1
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            self.metrics.error_count += 1
    
    async def _subscribe_to_symbols(self) -> None:
        """Subscribe to symbols for data streams"""
        if not self.websocket:
            return
        
        try:
            for symbol in self.config.symbols:
                subscribe_message = {
                    "action": "subscribe",
                    "symbol": symbol,
                    "type": self.config.stream_type.value
                }
                
                await self.websocket.send(json.dumps(subscribe_message))
                logger.info(f"Subscribed to {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to subscribe to symbols: {e}")
    
    def get_metrics(self) -> StreamMetrics:
        """Get current stream metrics"""
        if self.connection_start_time:
            self.metrics.uptime_seconds = (datetime.utcnow() - self.connection_start_time).total_seconds()
        return self.metrics


class StreamProcessor:
    """
    Processes and transforms stream data with filtering, aggregation, and validation.
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.processors: Dict[str, Callable] = {}
        self.filters: List[Callable] = []
        self.aggregators: Dict[str, Callable] = {}
        
        # Processing metrics
        self.messages_processed = 0
        self.messages_filtered = 0
        self.processing_time_ms = 0.0
        
        logger.info("StreamProcessor initialized")
    
    def add_processor(self, name: str, processor: Callable) -> None:
        """Add a data processor"""
        self.processors[name] = processor
        logger.info(f"Added processor: {name}")
    
    def add_filter(self, filter_func: Callable) -> None:
        """Add a data filter"""
        self.filters.append(filter_func)
        logger.info("Added data filter")
    
    def add_aggregator(self, name: str, aggregator: Callable) -> None:
        """Add a data aggregator"""
        self.aggregators[name] = aggregator
        logger.info(f"Added aggregator: {name}")
    
    async def process_event(self, event: StreamEvent) -> Optional[StreamEvent]:
        """Process a stream event"""
        start_time = time.time()
        
        try:
            # Apply filters
            for filter_func in self.filters:
                if not filter_func(event):
                    self.messages_filtered += 1
                    return None
            
            # Apply processors
            processed_data = event.data.copy()
            for name, processor in self.processors.items():
                try:
                    processed_data = await processor(processed_data)
                except Exception as e:
                    logger.error(f"Processor {name} failed: {e}")
                    continue
            
            # Apply aggregators
            aggregated_data = {}
            for name, aggregator in self.aggregators.items():
                try:
                    aggregated_data[name] = await aggregator(processed_data)
                except Exception as e:
                    logger.error(f"Aggregator {name} failed: {e}")
                    continue
            
            # Create processed event
            processed_event = StreamEvent(
                stream_type=event.stream_type,
                symbol=event.symbol,
                timestamp=event.timestamp,
                data={**processed_data, **aggregated_data},
                source=event.source,
                sequence_number=event.sequence_number
            )
            
            self.messages_processed += 1
            self.processing_time_ms = (time.time() - start_time) * 1000
            
            return processed_event
            
        except Exception as e:
            logger.error(f"Event processing failed: {e}")
            return None
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return {
            "messages_processed": self.messages_processed,
            "messages_filtered": self.messages_filtered,
            "average_processing_time_ms": self.processing_time_ms,
            "filter_ratio": self.messages_filtered / max(1, self.messages_processed + self.messages_filtered)
        }


class EventHandler:
    """
    Handles stream events with routing, priority queuing, and error handling.
    """
    
    def __init__(self):
        self.handlers: Dict[StreamType, List[Callable]] = {}
        self.priority_handlers: Dict[int, List[Callable]] = {}
        self.error_handlers: List[Callable] = []
        
        # Event queues
        self.high_priority_queue = Queue(maxsize=1000)
        self.normal_priority_queue = Queue(maxsize=10000)
        self.low_priority_queue = Queue(maxsize=5000)
        
        # Background tasks
        self._processing_tasks = []
        
        logger.info("EventHandler initialized")
    
    def add_handler(self, stream_type: StreamType, handler: Callable, priority: int = 1) -> None:
        """Add an event handler"""
        if stream_type not in self.handlers:
            self.handlers[stream_type] = []
        self.handlers[stream_type].append(handler)
        
        if priority not in self.priority_handlers:
            self.priority_handlers[priority] = []
        self.priority_handlers[priority].append(handler)
        
        logger.info(f"Added handler for {stream_type.value} with priority {priority}")
    
    def add_error_handler(self, handler: Callable) -> None:
        """Add an error handler"""
        self.error_handlers.append(handler)
        logger.info("Added error handler")
    
    async def start_processing(self) -> None:
        """Start event processing"""
        # Start processing tasks for each priority level
        self._processing_tasks = [
            asyncio.create_task(self._process_high_priority()),
            asyncio.create_task(self._process_normal_priority()),
            asyncio.create_task(self._process_low_priority())
        ]
        logger.info("Event processing started")
    
    async def stop_processing(self) -> None:
        """Stop event processing"""
        for task in self._processing_tasks:
            task.cancel()
        logger.info("Event processing stopped")
    
    async def handle_event(self, event: StreamEvent, priority: int = 1) -> None:
        """Handle a stream event"""
        try:
            # Route to appropriate queue based on priority
            if priority == 3:
                await self.high_priority_queue.put(event)
            elif priority == 1:
                await self.low_priority_queue.put(event)
            else:
                await self.normal_priority_queue.put(event)
                
        except Exception as e:
            logger.error(f"Failed to queue event: {e}")
            await self._handle_error(e, event)
    
    async def _process_high_priority(self) -> None:
        """Process high priority events"""
        while True:
            try:
                event = await self.high_priority_queue.get()
                await self._process_event(event, priority=3)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"High priority processing error: {e}")
    
    async def _process_normal_priority(self) -> None:
        """Process normal priority events"""
        while True:
            try:
                event = await self.normal_priority_queue.get()
                await self._process_event(event, priority=2)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Normal priority processing error: {e}")
    
    async def _process_low_priority(self) -> None:
        """Process low priority events"""
        while True:
            try:
                event = await self.low_priority_queue.get()
                await self._process_event(event, priority=1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Low priority processing error: {e}")
    
    async def _process_event(self, event: StreamEvent, priority: int) -> None:
        """Process an event with appropriate handlers"""
        try:
            # Call stream type handlers
            if event.stream_type in self.handlers:
                for handler in self.handlers[event.stream_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")
                        await self._handle_error(e, event)
            
            # Call priority handlers
            if priority in self.priority_handlers:
                for handler in self.priority_handlers[priority]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Priority handler error: {e}")
                        await self._handle_error(e, event)
                        
        except Exception as e:
            logger.error(f"Event processing error: {e}")
            await self._handle_error(e, event)
    
    async def _handle_error(self, error: Exception, event: Optional[StreamEvent] = None) -> None:
        """Handle processing errors"""
        for handler in self.error_handlers:
            try:
                await handler(error, event)
            except Exception as e:
                logger.error(f"Error handler failed: {e}")


class RealTimeStreamingSystem:
    """
    Main real-time streaming system that orchestrates WebSocket connections,
    stream processing, and event handling.
    """
    
    def __init__(self):
        self.streams: Dict[str, WebSocketManager] = {}
        self.processors: Dict[str, StreamProcessor] = {}
        self.event_handler = EventHandler()
        
        # System metrics
        self.total_messages = 0
        self.active_connections = 0
        self.system_start_time = datetime.utcnow()
        
        logger.info("RealTimeStreamingSystem initialized")
    
    async def add_stream(self, name: str, config: StreamConfig) -> bool:
        """Add a new data stream"""
        try:
            # Create WebSocket manager
            ws_manager = WebSocketManager(config)
            
            # Create stream processor
            processor = StreamProcessor(config)
            
            # Set up event flow
            ws_manager.on_message = self._handle_raw_message
            ws_manager.on_error = self._handle_connection_error
            ws_manager.on_connect = self._handle_connection_established
            ws_manager.on_disconnect = self._handle_connection_lost
            
            # Store components
            self.streams[name] = ws_manager
            self.processors[name] = processor
            
            # Connect to stream
            success = await ws_manager.connect()
            if success:
                self.active_connections += 1
            
            logger.info(f"Added stream: {name} ({'connected' if success else 'failed'})")
            return success
            
        except Exception as e:
            logger.error(f"Failed to add stream {name}: {e}")
            return False
    
    async def remove_stream(self, name: str) -> None:
        """Remove a data stream"""
        if name in self.streams:
            await self.streams[name].disconnect()
            del self.streams[name]
            del self.processors[name]
            self.active_connections -= 1
            logger.info(f"Removed stream: {name}")
    
    async def start(self) -> None:
        """Start the streaming system"""
        await self.event_handler.start_processing()
        logger.info("RealTimeStreamingSystem started")
    
    async def stop(self) -> None:
        """Stop the streaming system"""
        # Stop event processing
        await self.event_handler.stop_processing()
        
        # Disconnect all streams
        for name in list(self.streams.keys()):
            await self.remove_stream(name)
        
        logger.info("RealTimeStreamingSystem stopped")
    
    async def _handle_raw_message(self, event: StreamEvent) -> None:
        """Handle raw message from WebSocket"""
        self.total_messages += 1
        
        # Process through stream processor
        stream_name = event.source
        if stream_name in self.processors:
            processed_event = await self.processors[stream_name].process_event(event)
            if processed_event:
                # Route to event handler
                await self.event_handler.handle_event(processed_event)
    
    async def _handle_connection_error(self, error: str) -> None:
        """Handle connection errors"""
        logger.error(f"Connection error: {error}")
    
    async def _handle_connection_established(self) -> None:
        """Handle successful connection"""
        logger.info("Connection established")
    
    async def _handle_connection_lost(self) -> None:
        """Handle connection loss"""
        logger.warning("Connection lost")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        stream_metrics = {}
        for name, stream in self.streams.items():
            stream_metrics[name] = stream.get_metrics()
        
        processing_metrics = {}
        for name, processor in self.processors.items():
            processing_metrics[name] = processor.get_processing_metrics()
        
        uptime = (datetime.utcnow() - self.system_start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "total_messages": self.total_messages,
            "active_connections": self.active_connections,
            "stream_metrics": stream_metrics,
            "processing_metrics": processing_metrics
        }


# Example usage and configuration
async def main():
    """Example usage of the real-time streaming system"""
    
    # Create streaming system
    streaming_system = RealTimeStreamingSystem()
    
    # Configure streams
    market_data_config = StreamConfig(
        stream_type=StreamType.MARKET_DATA,
        url="wss://example.com/market-data",
        symbols=["EURUSD", "GBPUSD", "USDJPY"],
        heartbeat_interval=30,
        reconnect_delay=5,
        max_reconnect_attempts=10
    )
    
    # Add stream
    await streaming_system.add_stream("market_data", market_data_config)
    
    # Add event handlers
    async def handle_market_data(event: StreamEvent):
        print(f"Received market data for {event.symbol}: {event.data}")
    
    streaming_system.event_handler.add_handler(StreamType.MARKET_DATA, handle_market_data)
    
    # Start system
    await streaming_system.start()
    
    try:
        # Run for some time
        await asyncio.sleep(60)
    finally:
        await streaming_system.stop()


if __name__ == "__main__":
    asyncio.run(main()) 
