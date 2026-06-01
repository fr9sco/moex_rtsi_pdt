# Курсовая: прогноз IMOEX и RTSI с помощью PDT

Проект для курсовой работы по прогнозированию индексов Мосбиржи (`IMOEX`) и РТС (`RTSI`) на основе Permutation Decision Trees.

В текущей версии уже есть:

- загрузка дневных данных с MOEX ISS;
- EDA и базовые таблицы качества данных;
- собственная реализация `PermutationDecisionTreeClassifier`;
- признаки для временного ряда без подсматривания в будущее;
- разбиение `train / validation / test` по времени;
- walk-forward подбор гиперпараметров PDT;
- сравнение PDT с XGBoost, LSTM и простыми baseline-моделями

## Структура

```text
src/                  код пайплайна и моделей
scripts/              запуск этапов эксперимента
tests/                unit-тесты
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
```

## Проверки

```bash
python3 -m unittest discover -s tests
```

## Текущий результат

После walk-forward подбора PDT лучше всего проявился на задаче `RTSI, h=1`: test balanced accuracy `0.546367`

На остальных задачах простые бейзлайн-модели или LSTM остаются сильнее
