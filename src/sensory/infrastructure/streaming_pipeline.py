"""
Streaming Pipeline - Real-Time Data Processing Infrastructure

This module implements a production-ready streaming pipeline using Kafka and Redis
for real-time market data processing and distribution.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError

logger = logging.getLogger(__name__)

class StreamType(Enum):
    """Types of data streams"""
    MARKET_DATA = "market_data"
    ORDER_FLOW = "order_flow"
    ECONOMIC_DATA = "economic_data"
    NEWS_EVENTS = "news_events"
    ALERTS = "alerts"

@dataclass
class StreamMessage:
    """Standardized stream message format"""
    stream_type: StreamType
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    message_id: str

class StreamingPipeline:
    """
    Production streaming pipeline using Kafka and Redis
    Handles real-time data ingestion, processing, and distribution
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize streaming pipeline

        Args:
            config: Configuration with Kafka and Redis settings
        """
        self.config = config
        self.kafka_producer: Optional[AIOKafkaProducer] = None
        self.kafka_consumers: Dict[str, AIOKafkaConsumer] = {}
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Kafka settings
        self.bootstrap_servers = config.get('kafka_bootstrap_servers', ['localhost:9092'])
        self.topic_prefix = config.get('kafka_topic_prefix', 'market_intelligence')
        
        # Redis settings
        self.redis_url = config.get('redis_url', 'redis://localhost:6379')
        
        # Stream handlers
        self.stream_handlers: Dict[StreamType, List[Callable]] = {}
        
        # Performance metrics
        self.messages_processed = 0
        self.messages_failed = 0
        self.latency_stats = []

    async def initialize(self):
        """Initialize Kafka and Redis connections"""
        try:
            # Initialize Kafka producer
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            await self.kafka_producer.start()
            logger.info("Kafka producer initialized")

            # Initialize Redis client
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize streaming pipeline: {e}")
            raise

    async def shutdown(self):
        """Shutdown streaming pipeline"""
        try:
            # Close Kafka producer
            if self.kafka_producer:
                await self.kafka_producer.stop()
                logger.info("Kafka producer stopped")

            # Close Kafka consumers
            for consumer in self.kafka_consumers.values():
                await consumer.stop()
            self.kafka_consumers.clear()
            logger.info("Kafka consumers stopped")

            # Close Redis client
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis client closed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def publish_message(self, message: StreamMessage) -> bool:
        """
        Publish message to Kafka topic

        Args:
            message: Message to publish

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.kafka_producer:
                raise RuntimeError("Kafka producer not initialized")

            # Create topic name
            topic = f"{self.topic_prefix}.{message.stream_type.value}"
            
            # Prepare message
            message_dict = asdict(message)
            message_dict['timestamp'] = message.timestamp.isoformat()
            
            # Publish to Kafka
            await self.kafka_producer.send_and_wait(
                topic=topic,
                key=message.message_id,
                value=message_dict
            )
            
            # Update metrics
            self.messages_processed += 1
            
            logger.debug(f"Published message {message.message_id} to {topic}")
            return True

        except Exception as e:
            self.messages_failed += 1
            logger.error(f"Failed to publish message {message.message_id}: {e}")
            return False

    async def subscribe_to_stream(self, stream_type: StreamType, 
                                 handler: Callable[[StreamMessage], None]):
        """
        Subscribe to a data stream

        Args:
            stream_type: Type of stream to subscribe to
            handler: Function to handle incoming messages
        """
        try:
            if stream_type not in self.stream_handlers:
                self.stream_handlers[stream_type] = []
            
            self.stream_handlers[stream_type].append(handler)
            
            # Create consumer if not exists
            topic = f"{self.topic_prefix}.{stream_type.value}"
            if topic not in self.kafka_consumers:
                consumer = AIOKafkaConsumer(
                    topic,
                    bootstrap_servers=self.bootstrap_servers,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    group_id=f"{self.topic_prefix}_consumer_{stream_type.value}"
                )
                await consumer.start()
                self.kafka_consumers[topic] = consumer
                
                # Start consumer task
                asyncio.create_task(self._consume_messages(consumer, stream_type))
                
            logger.info(f"Subscribed to {stream_type.value} stream")

        except Exception as e:
            logger.error(f"Failed to subscribe to {stream_type.value}: {e}")
            raise

    async def _consume_messages(self, consumer: AIOKafkaConsumer, stream_type: StreamType):
        """Consume messages from Kafka topic"""
        try:
            async for message in consumer:
                try:
                    # Parse message
                    message_data = message.value
                    timestamp = datetime.fromisoformat(message_data['timestamp'])
                    
                    stream_message = StreamMessage(
                        stream_type=StreamType(message_data['stream_type']),
                        timestamp=timestamp,
                        data=message_data['data'],
                        source=message_data['source'],
                        message_id=message_data['message_id']
                    )
                    
                    # Call handlers
                    if stream_type in self.stream_handlers:
                        for handler in self.stream_handlers[stream_type]:
                            try:
                                handler(stream_message)
                            except Exception as e:
                                logger.error(f"Handler error for {stream_type.value}: {e}")
                    
                    # Update metrics
                    self.messages_processed += 1
                    
                except Exception as e:
                    self.messages_failed += 1
                    logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"Consumer error for {stream_type.value}: {e}")

    async def cache_data(self, key: str, data: Any, ttl: int = 3600):
        """
        Cache data in Redis

        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
        """
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")

            # Serialize data
            if isinstance(data, dict):
                serialized_data = json.dumps(data)
            else:
                serialized_data = str(data)

            # Store in Redis
            await self.redis_client.setex(key, ttl, serialized_data)
            logger.debug(f"Cached data with key: {key}")

        except Exception as e:
            logger.error(f"Failed to cache data {key}: {e}")

    async def get_cached_data(self, key: str) -> Optional[Any]:
        """
        Get cached data from Redis

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found
        """
        try:
            if not self.redis_client:
                raise RuntimeError("Redis client not initialized")

            # Get from Redis
            data = await self.redis_client.get(key)
            
            if data:
                # Try to deserialize as JSON first
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return data.decode('utf-8')
            
            return None

        except Exception as e:
            logger.error(f"Failed to get cached data {key}: {e}")
            return None

    async def publish_market_data(self, symbol: str, bid: float, ask: float, 
                                 volume: int, timestamp: datetime):
        """Publish market data to stream"""
        message = StreamMessage(
            stream_type=StreamType.MARKET_DATA,
            timestamp=timestamp,
            data={
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'volume': volume,
                'spread': ask - bid
            },
            source='price_feed',
            message_id=f"md_{symbol}_{timestamp.timestamp()}"
        )
        
        await self.publish_message(message)

    async def publish_order_flow(self, symbol: str, order_book: Dict[str, Any], 
                                timestamp: datetime):
        """Publish order flow data to stream"""
        message = StreamMessage(
            stream_type=StreamType.ORDER_FLOW,
            timestamp=timestamp,
            data={
                'symbol': symbol,
                'order_book': order_book
            },
            source='order_flow_feed',
            message_id=f"of_{symbol}_{timestamp.timestamp()}"
        )
        
        await self.publish_message(message)

    async def publish_economic_data(self, indicator: str, value: float, 
                                   timestamp: datetime):
        """Publish economic data to stream"""
        message = StreamMessage(
            stream_type=StreamType.ECONOMIC_DATA,
            timestamp=timestamp,
            data={
                'indicator': indicator,
                'value': value
            },
            source='fred_api',
            message_id=f"ed_{indicator}_{timestamp.timestamp()}"
        )
        
        await self.publish_message(message)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'messages_processed': self.messages_processed,
            'messages_failed': self.messages_failed,
            'success_rate': (self.messages_processed / (self.messages_processed + self.messages_failed)) 
                           if (self.messages_processed + self.messages_failed) > 0 else 0,
            'active_consumers': len(self.kafka_consumers),
            'active_handlers': sum(len(handlers) for handlers in self.stream_handlers.values())
        }

# Example usage
async def example_streaming_pipeline():
    """Example of using the streaming pipeline"""
    
    # Configuration
    config = {
        'kafka_bootstrap_servers': ['localhost:9092'],
        'kafka_topic_prefix': 'market_intelligence',
        'redis_url': 'redis://localhost:6379'
    }
    
    # Initialize pipeline
    pipeline = StreamingPipeline(config)
    await pipeline.initialize()
    
    # Define message handler
    def market_data_handler(message: StreamMessage):
        print(f"Received market data: {message.data}")
    
    # Subscribe to market data stream
    await pipeline.subscribe_to_stream(StreamType.MARKET_DATA, market_data_handler)
    
    # Publish some test data
    await pipeline.publish_market_data('EURUSD', 1.1000, 1.1001, 1000, datetime.now())
    
    # Wait a bit
    await asyncio.sleep(1)
    
    # Get metrics
    metrics = pipeline.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    # Shutdown
    await pipeline.shutdown()

if __name__ == "__main__":
    asyncio.run(example_streaming_pipeline()) 