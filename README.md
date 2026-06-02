# Курсовая: прогноз IMOEX и RTSI с помощью PDT

Проект для курсовой работы по прогнозированию индексов Мосбиржи (`IMOEX`) и РТС (`RTSI`) на основе Permutation Decision Trees.

В текущей версии уже есть:

- загрузка дневных данных с MOEX ISS;
- EDA и базовые таблицы качества данных;
- собственная реализация `PermutationDecisionTreeClassifier`;
- признаки для временного ряда без подсматривания в будущее;
- разбиение `train / validation / test` по времени;
- walk-forward подбор гиперпараметров PDT;
- сравнение PDT с XGBoost, LSTM и простыми baseline-моделями;
- простой backtest стратегии по прогнозному направлению.

## Структура

```text
src/                  код пайплайна и моделей
scripts/              запуск этапов эксперимента
tests/                unit-тесты
backend/              FastAPI для dashboard
frontend/             React dashboard
reports/figures/      небольшие графики для отчета
reports/tables/       итоговые таблицы экспериментов
reports/trees/        текстовые выгрузки PDT-деревьев
Литература/           статьи и конспекты
Этап_*.md             заметки по этапам курсовой
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

На macOS для XGBoost может понадобиться:

```bash
brew install libomp
```

## Запуск этапов

```bash
python3 scripts/stage1_data_eda.py --from-date 2015-01-01 --till-date 2026-05-28
python3 scripts/stage2_pdt_sanity.py
python3 scripts/stage3_build_datasets.py --horizons 1 5
python3 scripts/stage3_pdt_pipeline.py --horizons 1 5
python3 scripts/stage4_model_comparison.py --horizons 1 5 --lstm-max-epochs 35 --lstm-patience 6
python3 scripts/stage5_backtest.py --cost-bps 5
```

## Проверки

```bash
python3 -m unittest discover -s tests
```

## Веб-панель

Backend:

```bash
python3 -m uvicorn backend.main:app --reload --port 8000
```

Для графиков прогнозов и equity нужны файлы `reports/tables/stage4_predictions.csv` и `reports/tables/stage5_backtest_equity.csv`. Если их нет локально, сначала запусти этапы 4 и 5.

Frontend:

```bash
cd frontend
npm install
npm run dev
```

После запуска страница доступна по адресу `http://127.0.0.1:5173`.

## Текущий результат

После walk-forward подбора PDT лучше всего проявился на задаче `RTSI, h=1`: test balanced accuracy `0.546367`

На остальных задачах простые бейзлайн-модели или LSTM остаются сильнее.

Backtest на test-периоде использует простую стратегию `long/cash`: если модель прогнозирует рост, держим индекс до горизонта прогноза, иначе сидим в кэше. Для `h=5` берем неперекрывающиеся сделки

На `RTSI, h=1` стратегия PDT дала total return `0.270448` против `0.072432` у buy-and-hold за test-период. На пятидневном горизонте PDT пока не дает устойчивого результата
