# 🐺 Evolving Market Predator (EMP) v1.0

**An Autonomous Hunter-Killer Algorithm for Financial Markets**

![Status](https://img.shields.io/badge/status-v1.0_Apex-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Architecture](https://img.shields.io/badge/architecture-v1.1-clean-blue.svg)

## 🎯 What is the Evolving Market Predator?

The EMP is **not** a trading bot—it's a **digital alpha predator**. Operating on Darwinian evolution principles, it uses a powerful Genetic Programming engine to breed apex trading strategies that hunt for alpha in the chaos of modern markets.

Within its **Simulation Envelope**—a digital jungle where only the fittest survive—primitive strategies compete, mutate, crossbreed, and adapt over hundreds of generations. The weakest are culled. The strongest become the next generation of hunters.

## 🧬 The Predator's Anatomy (Architecture v1.1)

The system is built on a strict 8-layer modular architecture designed for evolutionary adaptability:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          7. GOVERNANCE LAYER                            │
│                    (The Rules of Engagement)                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ Defines "Good" & Audits
┌───────────────────────────────┴─────────────────────────────────────────┐
│                    8. OPERATIONAL BACKBONE                               │
│              (Metabolism & Nervous System)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐                │
│  │ StateStore   │──│ EventBus     │──│ Container      │                │
│  │ (Redis)      │  │ (NATS)       │  │ Orchestrator   │                │
│  └──────────────┘  └──────────────┘  └────────────────┘                │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────────────┐
│  1. SENSORY LAYER          │  2. THINKING LAYER      │  6. TRADING LAYER │
│  "What is the scent?"      │  "Prey or danger?"      │  "How to strike?" │
└──────────────┬─────────────┴──────────────┬──────────┴──────────────┬────┘
               │                           │                          │
               └───────────────────────────┼──────────────────────────┘
                                           │
                               ┌───────────┴───────────┐
                               │ 5. ADAPTIVE CORE      │
                               │ (Evolutionary Heart)  │
                               └───────────┬───────────┘
                                           │
                               ┌───────────┴───────────┐
                               │ 3. SIMULATION ENVELOPE│
                               │   (Digital Jungle)    │
                               └───────────────────────┘
```

### Layer Responsibilities

| Layer | Mission |
|-------|---------|
| **1. Sensory Layer** | "What is the scent of the market right now?" |
| **2. Thinking Layer** | "Does this scent mean prey or danger?" |
| **3. Simulation Envelope** | "Can this organism survive the jungle?" |
| **4. UI Layer** | "How does the handler observe and command?" |
| **5. Adaptive Core** | "How do we forge a superior hunter?" |
| **6. Trading Layer** | "How does the predator strike with precision?" |
| **7. Governance Layer** | "What are the genetic rules of engagement?" |
| **8. Operational Backbone** | "What is the predator's metabolism?" |

## 🏹 The Anatomy of a Hunt

The predator's actions follow a high-speed, asynchronous cycle:

1. **The Scent**: CTraderDataOrgan detects market movement → `MarketUnderstanding` event
2. **The Stalk**: ThinkingManager analyzes context → `ContextPacket` 
3. **The Kill Decision**: AdaptiveCore processes with champion genome → `TradeIntent`
4. **The Final Check**: RiskGateway validates against rules → Approved/Rejected
5. **The Strike**: TradingManager executes via CTraderBrokerInterface
6. **Confirmation & Learning**: ExecutionEvent updates state → Learning occurs

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Core** | Python 3.10+, asyncio |
| **Architecture** | NATS (Event Bus), Docker |
| **Data** | Redis (State), SQLite (Governance), FAISS (Memory) |
| **API** | FastAPI, WebSockets |
| **CLI** | Typer |
| **Quality** | Ruff, MyPy, pytest |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Git

### Step 1: Clone & Setup
```bash
git clone https://github.com/HWeber-tech/emp_proving_ground_v1.git
cd emp_proving_ground_v1
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Start Infrastructure
```bash
docker-compose up -d
```

### Step 3: Configure Environment
```bash
cp .env.example .env
# Edit .env with your cTrader credentials for live mode
```

### Step 4: Initialize System
```bash
# Initialize governance database
python -m src.governance.strategy_registry --init-db

# Download market data
python -m src.cli.main data download EURUSD
```

### Step 5: Run the Predator
```bash
# Start evolution
python -m src.cli.main run evolution

# Run simulation
python -m src.cli.main run simulation --strategy-id <ID>

# Live paper trading
python -m src.cli.main run live
```

## 📊 Project Status

### ✅ v1.0 - Apex Specimen Complete
- [x] v1.1 Architecture fully realized
- [x] Event-driven core with NATS
- [x] High-fidelity Simulation Envelope
- [x] Genetic Programming engine
- [x] Stateful Trading Layer with RiskGateway
- [x] Governance Layer with StrategyRegistry
- [x] Real-time monitoring via WebSocket
- [x] IC Markets cTrader Demo API integration

### 🎯 v1.1 - Production Hardening (Next)
- [ ] Token refreshing (OPS-04)
- [ ] Dynamic symbol mapping (SENSORY-04)
- [ ] Structured logging (OPS-05)
- [ ] Advanced fitness functions (GOV-04)
- [ ] Handler's Dashboard (UI-03)

### 🔮 v2.0 - Cognitive Enhancement (Future)
- [ ] Real-time web dashboard
- [ ] Sensory expansion (order book, sentiment)
- [ ] Active instinct with PatternMemory
- [ ] Meta-evolution capabilities

## 🎮 CLI Commands

The predator CLI is your main control interface:

```bash
# Evolution management
predator evolution start --generations 100
predator evolution status
predator evolution pause

# Strategy management
predator strategy list
predator strategy deploy <ID>
predator strategy retire <ID>

# Live trading
predator live start --account demo
predator live status
predator live stop

# Monitoring
predator monitor dashboard
predator monitor logs --tail 100
```

## 🔧 Configuration

### Environment Variables
```bash
# cTrader API
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_client_secret
CTRADER_ACCESS_TOKEN=your_access_token
CTRADER_ACCOUNT_ID=your_account_id

# Redis
REDIS_URL=redis://localhost:6379

# NATS
NATS_URL=nats://localhost:4222
```

### Configuration Files
- `config.yaml`: Main system configuration
- `config/governance/strategy_registry.yaml`: Strategy governance
- `config/fitness/default_v1.yaml`: Fitness function parameters

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/unit/
pytest tests/integration/
pytest tests/end_to_end/

# Run with coverage
pytest --cov=src tests/
```

## 📈 Monitoring

### Real-time Dashboard
Access the WebSocket dashboard at `http://localhost:8000/dashboard`

### Metrics Endpoints
- Health: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`
- WebSocket: `ws://localhost:8000/ws`

## 🐳 Docker Deployment

```bash
# Development
docker-compose up

# Production
docker-compose -f docker-compose.prod.yml up -d
```

## 🤝 Contributing

1. Ensure code passes `ruff` and `mypy`
2. Write tests for new features
3. Update documentation
4. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This is experimental software for educational purposes. Use at your own risk. Always test thoroughly in simulation before live trading.

---

**The Evolving Market Predator** - Where Darwin meets Wall Street.
