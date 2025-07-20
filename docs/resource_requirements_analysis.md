# Resource Requirements Analysis - Phase 2 Components

## What Makes These Systems "Heavy"?

### 1. **5D+1 Sensory Cortex Processing**
The "heavy" resources come from these specific computational demands:

#### **WHY Dimension - Macro Predator Intelligence**
- **NLP Processing**: Real-time analysis of central bank statements
  - BERT/RoBERTa models: ~500MB-2GB RAM per model
  - Sentiment analysis: ~100-500MB per document batch
  - Multi-language support: ~1-2GB additional RAM

- **Geopolitical Analysis**: 
  - News feed processing: ~50-100MB per hour of global news
  - Event correlation: ~200-500MB for historical pattern matching
  - Risk scoring: ~100MB for real-time calculations

#### **HOW Dimension - Institutional Footprint Hunter**
- **ICT Pattern Detection**:
  - Order block analysis: ~10-50MB per 1000 bars
  - Fair value gap detection: ~5-20MB per timeframe
  - Liquidity sweep analysis: ~20-100MB per session
  - Volume profile: ~50-200MB for full market depth

- **Smart Money Flow**:
  - Real-time order flow: ~100-500MB per hour
  - Institutional footprint: ~200MB for historical analysis
  - Cross-asset correlation: ~500MB-1GB for multi-market analysis

### 2. **Advanced Evolution Engine**
- **Genetic Algorithm Processing**:
  - Population storage: ~10-100MB per 1000 genomes
  - Fitness evaluation: ~1-5GB for backtesting across regimes
  - Stress testing: ~2-10GB for scenario simulation
  - Memory for strategy variants: ~500MB-2GB

- **Multi-dimensional Fitness**:
  - Historical data: ~1-5GB per asset class
  - Performance metrics: ~100-500MB per generation
  - Correlation matrices: ~100MB-1GB depending on universe size

### 3. **Real-time Data Processing**
- **Market Data Ingestion**:
  - Tick data: ~1-5GB per day per major pair
  - Order book snapshots: ~10-50GB per day for full depth
  - News feeds: ~100MB-1GB per day
  - Economic calendar: ~10-50MB per day

### 4. **Memory vs CPU vs Storage Breakdown**

#### **Memory Intensive Components**:
- **NLP Models**: 500MB-2GB (WHY dimension)
- **Historical Data**: 1-10GB (Evolution engine)
- **Real-time Buffers**: 100MB-1GB (Sensory cortex)

#### **CPU Intensive Components**:
- **Pattern Recognition**: High CPU for complex calculations
- **Genetic Algorithms**: CPU-intensive for fitness evaluation
- **Real-time Correlation**: CPU-heavy for matrix operations

#### **Storage Intensive Components**:
- **Historical Data**: 10-100GB for multi-year datasets
- **Backtesting Results**: 1-10GB per strategy generation
- **Log Aggregation**: 100MB-1GB per day

## **Actual Resource Requirements by Use Case**

### **Development/Testing** (~$15-25/month)
```
CPU: 1-2 cores
RAM: 2-4GB
Storage: 20-50GB
Network: 1-5 Mbps
```

### **Small Production** (~$60-90/month)
```
CPU: 2-4 cores  
RAM: 4-8GB
Storage: 50-100GB
Network: 10-50 Mbps
```

### **Full Production** (~$200-500/month)
```
CPU: 8-16 cores
RAM: 16-32GB
Storage: 100-500GB
Network: 100+ Mbps
```

## **Optimization Strategies**

### **1. Lazy Loading**
- Load NLP models only when needed
- Process data in batches vs real-time
- Cache frequently used patterns

### **2. Sampling**
- Use 1-minute bars instead of tick data
- Sample news feeds vs full firehose
- Limit historical depth for evolution

### **3. Compression**
- Compress historical data (10:1 ratio)
- Use efficient data structures
- Binary formats vs JSON

### **4. Tiered Processing**
- **Tier 1**: Critical real-time (1-5 second latency)
- **Tier 2**: Important near-real-time (1-5 minute latency)  
- **Tier 3**: Background analysis (hourly/daily)

### **5. Resource Sharing**
- Shared Redis cache across components
- Common historical data storage
- Reusable computation results

## **Minimal Viable Setup**

### **Single t3.small instance can handle:**
- ✅ 1-2 currency pairs
- ✅ 15-minute timeframe analysis
- ✅ 30-day historical depth
- ✅ Basic pattern recognition
- ✅ Simple evolution (100-500 genomes)

### **What gets sacrificed with minimal resources:**
- ❌ Real-time tick analysis
- ❌ Multi-asset correlation
- ❌ Deep historical backtesting
- ❌ Complex NLP models
- ❌ High-frequency evolution

## **Scaling Triggers**

### **Scale UP when:**
- CPU >70% for 5+ minutes
- Memory >80% for 5+ minutes
- Response time >2 seconds
- Data volume >10GB/day

### **Scale DOWN when:**
- CPU <20% for 30+ minutes
- Memory <40% for 30+ minutes
- Response time <500ms
- Data volume <1GB/day

## **Bottom Line Resource Needs**

### **For Phase 2 MVP:**
- **Single t3.small**: $15-25/month
- **Handles**: Basic 5D+1 functionality
- **Limitations**: Reduced accuracy, slower processing
- **Upgrade path**: Scale up as profitable

### **For Full Accuracy:**
- **t3.medium cluster**: $60-90/month
- **Handles**: Full 5D+1 with good accuracy
- **Limitations**: None for most use cases

The "heavy" resources are primarily for **enterprise-grade accuracy and scale** - but the system works perfectly well on modest hardware for development and small-scale production.
