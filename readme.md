# 🤖 Trading Bot — MTF ML Strategy

Мульти-таймфреймовый торговый бот на Python с ML-моделью (RandomForest), фильтром режима рынка (1d SMA200), риск-менеджментом и честным бэктестом (walk-forward + out-of-sample + издержки).

> ⚠️ **Статус: research / paper trading.** Бот не гарантирует прибыль.
> Перед реальными деньгами — обязательный Testnet и минимальный депозит.

---

## Что умеет

- **Данные:** скачивание 1h / 4h / 1d OHLCV с биржи через `ccxt` (Binance, Bybit, OKX).
- **Фичи:** мульти-таймфрейм — RSI, MACD, SMA, волатильность, объём (1h + 4h).
- **ML:** `RandomForestClassifier` предсказывает значимое движение цены (>0.5%).
- **Валидация:** walk-forward (5 фолдов) + holdout — честная оценка обобщения.
- **Бэктест:** комиссии, slippage, SL/TP, трейлинг-стоп, min_hold, фильтр режима.
- **Режим рынка:** торгуем только когда цена выше SMA200 на 1d (bull regime).
- **Риск-менеджмент:** размер позиции от баланса, стоп-лосс, тейк-профит.
- **DRY_RUN:** виртуальные сделки, запись в SQLite, PnL.
- **Тесты:** `pytest` (8 тестов), без сетевых вызовов (моки/синтетика).
- **Docker:** готовый образ для облачного развёртывания.

---

## Архитектура

```
trading-bot/
├── src/
│   ├── core/config.py          # Конфиг из .env
│   ├── data/collector.py       # (опционально) сбор данных
│   ├── analysis/
│   │   ├── mtf_features.py     # 1h/4h индикаторы + merge_asof (без look-ahead)
│   │   ├── strategy.py         # MLStrategy (legacy, 1h-only)
│   │   └── rule_strategy.py    # RuleStrategy (mean-reversion, альтернатива)
│   ├── risk/manager.py         # Размер позиции, SL/TP
│   ├── execution/executor.py   # Отправка ордеров (DRY_RUN / real)
│   ├── database/models.py      # SQLAlchemy: Trade
│   └── main.py                 # Главный цикл (MTF + regime)
├── scripts/
│   ├── download_data.py        # Скачивание 1h/4h/1d
│   ├── train_model_mtf.py      # Обучение MTF-модели + walk-forward
│   └── backtest_mtf.py         # Бэктест с издержками и SL/TP
├── tests/
│   ├── test_mtf_features.py    # Проверка индикаторов и отсутствия look-ahead
│   ├── test_strategy.py        # Legacy MLStrategy
│   └── test_risk.py            # RiskManager
├── data/                       # CSV с OHLCV (gitignore)
├── models/                     # model_mtf.joblib (gitignore)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── conftest.py                 # sys.path для pytest
├── .env.example
└── README.md
```

**Поток данных:**
Биржа (ccxt) → download_data.py → CSV (data/)
│
▼
train_model_mtf.py → model_mtf.joblib
│
▼
backtest_mtf.py (историческая проверка)
│
▼
main.py (DRY_RUN → Testnet → Real)

---

## Установка

### 1. Клонирование и окружение

```bash
git clone <repo> trading-bot
cd trading-bot
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`:

```
ccxt>=4.1.0
pandas>=2.1.0
scikit-learn>=1.3.0
SQLAlchemy>=2.0.0
python-dotenv>=1.0.0
loguru>=0.7.0
joblib>=1.3.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

### 2. Конфиг

Скопируйте `.env.example` → `.env` и заполните:

```
EXCHANGE_ID=binance
API_KEY=
API_SECRET=
SYMBOL=ETH/USDT
TIMEFRAME=1h
DRY_RUN=True
DB_URL=sqlite:///trading.db
DATA_DIR=./data
MODEL_DIR=./models
MAX_POSITION_SIZE_PCT=0.05
STOP_LOSS_PCT=0.03
TAKE_PROFIT_PCT=0.06
```

> `API_KEY` / `API_SECRET` — не нужны для скачивания данных и `DRY_RUN` (публичные эндпоинты).
> Нужны только для Testnet / реальной торговли.

---

## Быстрый старт

### Шаг 1. Скачать данные

```bash
python -m scripts.download_data 1h
python -m scripts.download_data 4h
python -m scripts.download_data 1d
```

В `data/` появятся `ETH_USDT_1h.csv`, `ETH_USDT_4h.csv`, `ETH_USDT_1d.csv`.

### Шаг 2. Обучить модель

```bash
python -m scripts.train_model_mtf
```

Вывод: walk-forward accuracy по 5 фолдам, holdout accuracy, feature importance.
Модель сохраняется в `models/model_mtf.joblib`.

**Ориентир:** walk-forward accuracy >52% ± <5% — есть слабый edge.
Если accuracy ~50% ± >8% — модель не нашла паттерн, стратегия не торгуемая.

### Шаг 3. Бэктест

```bash
# Out-of-sample (последние 20%) с издержками, SL/TP, фильтром режима
python -m scripts.backtest_mtf

# Вся история (in-sample) — для сравнения
python -c "from scripts.backtest_mtf import backtest_mtf; backtest_mtf(start_idx=250)"

# Без фильтра режима
python -c "from scripts.backtest_mtf import backtest_mtf; backtest_mtf(use_regime_filter=False)"
```

**Метрики в выводе:**

- `Profit %` — итоговая доходность
- `Win rate` — доля прибыльных сделок
- `Exit breakdown` — как закрывались позиции (SL, TP, TRAIL, SELL)
- `Buy & Hold` — бенчмарк

**Цель:** `profit ≥ Buy & Hold`, `win rate >50%`, `trades 20–80`.

### Шаг 4. Тесты

```bash
pytest tests/ -v
```

Ожидаемо: `8 passed`.

### Шаг 5. DRY_RUN

В `src/main.py`:

```python
TEST_MODE = True   # режим B: ослабленные условия (проверка механики)
# TEST_MODE = False  # режим A: строгие условия (реальная торговля)
```

Запуск:

```bash
python -m src.main
```

- **Режим B** — бот торгует часто (проверка механики: BUY/SELL, запись в БД, PnL).
- **Режим A** — бот ждёт строгого сигнала (`P(up)>0.58 + trend_4h=1 + regime=BULL`).

**Проверка БД:**

```bash
python -c "import sqlite3; c=sqlite3.connect('trading.db'); [print(r) for r in c.execute('SELECT * FROM trades').fetchall()]"
```

**Остановка:** `Ctrl+C`.

---

## Docker

### Сборка и запуск

```bash
docker-compose up --build -d
docker-compose logs -f bot
```

`docker-compose.yml` (пример):

```yaml
version: "3.8"
services:
  bot:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./trading.db:/app/trading.db
    restart: unless-stopped
```

`Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "src.main"]
```

> Важно: перед сборкой образа данные и модель должны быть уже скачаны/обучены локально (они монтируются через volumes).

---

## Переход на Testnet / реальные деньги

### Testnet Binance

Зарегистрируйтесь на https://testnet.binance.vision — получите API-ключи.

В `.env`:

```
DRY_RUN=False
API_KEY=your_testnet_key
API_SECRET=your_testnet_secret
EXCHANGE_ID=binance
```

В `src/main.py` (в `main()`, после создания exchange):

```python
exchange.set_sandbox_mode(True)
```

Запустите `python -m src.main`. Сделки будут реальными в рамках Testnet.

### Реальные деньги

Только после 30+ сделок на Testnet с положительным PnL.

- Минимальный депозит: $50–100.
- `DRY_RUN=False`, `set_sandbox_mode(False)`, реальные ключи с правами только на торговлю (без вывода).
- Мониторинг: логи + БД + баланс на бирже.

---

## Известные ограничения

- **Long-only.** Стратегия не шортит. В падающем рынке — в кэше (HOLD) или теряет меньше рынка.
- **Один символ.** Не портфельная. Для нескольких пар — запускать несколько инстансов с разными `.env`.
- **Нет funding rate.** Для фьючерсов (если добавите шорты) нужно учитывать плату за удержание.
- **Нет order book.** Только OHLCV. Для HFT/скальпинга не подходит.
- **SQLite.** Для production — PostgreSQL (поменять `DB_URL`).
- **Нет Telegram-алертов.** Логи только в файл/консоль. Легко добавить `loguru` sink.
- **Модель не переобучается автоматически.** При смене режима рынка — переобучить вручную (`train_model_mtf`).

---

## Roadmap

- [x] Данные 1h/4h/1d, MTF-фичи
- [x] Walk-forward + holdout валидация
- [x] Бэктест с комиссиями, slippage, SL/TP, trailing
- [x] Фильтр режима рынка (1d SMA200)
- [x] DRY_RUN + SQLite + PnL
- [x] 8/8 тестов, Docker
- [ ] Шорт-сигналы (фьючерсы, Bybit/Binance Futures)
- [ ] Testnet Binance / Bybit (1–2 недели)
- [ ] Telegram-алерты
- [ ] PostgreSQL вместо SQLite
- [ ] Портфель из 3–5 пар
- [ ] Автоматическое переобучение раз в неделю

---

## Дисклеймер

Это исследовательский проект. Торговля криптовалютами связана с высоким риском потери капитала. Автор не несёт ответственности за финансовые потери. Используйте на свой страх и риск. Никогда не торгуйте деньгами, которые не готовы потерять.

## Лицензия

MIT License. Используйте на свой страх и риск.
