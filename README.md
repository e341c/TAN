# TAN
```
quant-regressor/
├── deploy/
│   ├── docker-compose.yml
│   ├── k8s/
│   │   ├── go-infer.yaml
│   │   ├── trainer.yaml
│   │   └── registry.yaml
├── infer/                        # инференс + сигналы + исполнение
│   ├── cmd/main.go
│   ├── internal/
│   │   ├── datafeed/             # WS биржи, ClickHouse writer
│   │   ├── features/             # RSI/MACD/ATR/z-score/σ, кэш окон
│   │   ├── model/                # onnxruntime wrapper, hot-reload
│   │   ├── signal/               # thresholds, vol targeting
│   │   ├── risk/                 # SL/TP, лимиты, cooldown
│   │   ├── exec/                 # ордера, ретраи, отчёты
│   │   ├── api/                  # REST/gRPC (signals, health, metrics)
│   │   └── store/                # ClickHouse, Postgres
│   └── config/
│       └── config.yaml
├── technical-analysis-trainer/   # обучение
│   ├── train.py                  # основной pipeline
│   ├── features.py               # расчёт фич оффлайн
│   ├── dataset.py                # выборки, split, leakage guard
│   ├── model.py                  # LGBMRegressor + early stopping
│   ├── export_onnx.py            # сохранение в ONNX + метаданные
│   ├── eval.py                   # метрики, репорт, PnL sim
│   └── trainer_api.py            # (опц) REST: /train, /status, /latest
├── model-registry/               # S3/MinIO или простой REST-сторож
│   ├── app.py                    # выдаёт версии/артефакты
│   └── storage/                  # {version}/model.onnx, meta.json
└── shared/
    ├── schemas/                  # Avro/JSON схемы сообщений/метаданных
    └── docs/                     # ТЗ, ADR, SLA
```

## Обучение и протокол оценки

Оба тренера используют walk-forward фолды, scaler на train, ансамбль по seed'ам
и обязательные baseline'ы. Метрика без baseline ничего не значит.

```bash
# Quantile: 3 фолда x 5 seeds, conformal-калибровка, HAR-RV benchmark
python -m technical_analysis_trainer.models.quantile.train

# Trend: triple-barrier таргет, ансамбль + GBM baseline
python -m technical_analysis_trainer.models.trend.train_v2

# Какой прогон деплоить — по val, не по test
python -m technical_analysis_trainer.models.trend.select_run

# Скользящая оценка: модель на каждый месяц, оценка на следующем
python -m technical_analysis_trainer.apps.rolling_eval --months 36 --seeds 5
# Её результаты как интерактивный отчёт Plotly (без переобучения)
python -m technical_analysis_trainer.apps.rolling_report --json artifacts/rolling_eval.json

# Permutation importance по блокам фич (на val)
python -m technical_analysis_trainer.apps.feature_importance

# Проверка протокола (splits, embargo, conformal, HAR)
python -m technical_analysis_trainer.core.test_validation
```

**Артефакт trend — не один из фолдов.** Фолды нужны, чтобы *оценить* качество и
выбрать конфигурацию; модель, которая едет в прод, дообучается на всём
разрешённом периоде (`train_final` в `train_v2.py`). Раньше экспортировался
`folds[-1]`, и у прогона 2026-08-17 это означало веса и scaler, обученные по
данным до 2025-05-02 — шестнадцать месяцев отставания в проде. Валидационного
блока у финальной модели нет намеренно: бюджет эпох берётся как медиана
`kept_epochs` фолдов, а освободившиеся бары уходят в обучение и калибровку.

**Калибровка.** На отложенном срезе (~8% хвоста обучающего периода) обучается
Platt-поправка и подбирается порог решения; оба лежат в бандле артефакта и
применяются и в проде, и в бэктесте. Важно: монотонная поправка **не меняет
AUC** — она чинит решение, а не ранжир, и подавать её как улучшение модели
нельзя.

**Якорь барьера.** `barrier_anchor: last_observed` — барьеры считаются от
закрытия последнего бара в окне, а не следующего. Старая схема отдавала модели
цель, отсчитанную от цены, которой она не видела: на BTC 1h один бар шума =
43.9% полуширины барьера 2.5σ, а в проде якорный бар вообще ещё не закрылся.
Конвенция пишется в `target_spec` артефакта, старые артефакты читаются как
`next_close`.

**Доверительные интервалы — блочные.** `core/metrics.block_bootstrap_ci`
ресемплит непрерывные блоки длиной `max_holding`, потому что triple-barrier
метки перекрываются (медианный span 6 баров при 24). i.i.d.-бутстрап давал
интервал примерно в √(1/uniqueness) ≈ 2.3 раза уже, чем нужно. Рядом с `n`
печатается эффективное `n` (`core/metrics.effective_n`).

### Правила, зашитые в код

**Выбор конфигурации — только по val.** `core.validation.select_by_val`
выбрасывает исключение при попытке выбрать по test-метрике. Причина: в старом
`train_runs.csv` 184 прогона дали ROC-AUC 0.5305 ± 0.0052 при нулевой
корреляции с любым гиперпараметром (`hidden_size`: +0.000), а лучший прогон
(0.5457) в точности совпадает с ожидаемым максимумом 184 шумовых выборок
(0.5439). Выбор лучшего по test завышает метрику примерно на +2.5σ — в
несколько раз больше реального эджа.

**Embargo между сплитами.** Метки смотрят вперёд на `horizon` (или
`max_holding` для triple barrier) баров; без разрыва последние train-метки
залезают в val.

**Ансамбль вместо лотереи seed'ов.** Разброс между seed'ами доминирует, поэтому
модели усредняются (`QuantileEnsemble`, `TrendEnsemble`) — обе экспортируются в
один ONNX-граф.

### Что показывают метрики

| Модель | Есть эдж | Нет эджа |
|---|---|---|
| Quantile | хвосты: pinball skill **+4.4% против HAR-RV**, покрытие 90.3% при номинале 90% | медиана: skill **+0.0003**, MAE не бьёт persistence |
| Trend | triple barrier 2.5σ: val AUC 0.5237 | next_bar: AUC 0.5102; barrier 1.0σ: 0.4937 (хуже монетки) |

Это метрики held-out фолда **внутри** периода обучения. На свечах, которых модель
не видела вообще, картина другая — см. [Holdout](#holdout-свечи-которых-модели-не-видели).

Ширина барьера выбрана по **val** AUC: 1.75σ→0.5141, **2.5σ→0.5237**,
3.5σ→0.5122. По test выиграл бы 3.5σ (0.5390) — это ровно та ошибка, от которой
защищает правило выше.

## Тесты

```bash
# протокол обучения: splits, embargo, conformal, HAR — синтетика, без данных
python -m technical_analysis_trainer.core.test_validation
# метрики живых предсказаний — синтетика, без данных
python -m technical_analysis_trainer.storage.test_prediction_metrics
# спаны обучения/оценки выровнены по барам, а не по строкам — на train-данных
python -m technical_analysis_trainer.models.trend.test_spans
# batch-скоринг бэктеста == live-путь          ─┐ гоняются по holdout-свечам,
python -m technical_analysis_trainer.apps.test_backtest             #  │ докачивают данные
# backfill не подглядывает в будущее           ─┘ при устаревшем кэше
python -m technical_analysis_trainer.apps.test_backfill_predictions

cd trading_bot && go test ./...
```

Два последних набора качают holdout-данные, если кэш протух; `HOLDOUT_OFFLINE=1`
запрещает сеть и роняет тест вместо докачки. `models/trend/test_cli.py` — это не
тест, а инструмент визуализации v1-модели: он оценивает **next-bar sign**, а
задеплоенная модель триплбарьерная, так что его accuracy про неё ничего не
говорит.

## Граница обучение / оценка

Одна дата в [`assembly/config.yaml`](technical_analysis_trainer/assembly/config.yaml)
делит данные надвое:

```yaml
frozen_eval_start: "2026-08-01T00:00:00Z"
```

Всё до неё доступно обучению, всё с неё — только оценке. `core/protocol.py`
читает эту дату в начале тренировки, отрезает хвост **до расчёта любой фичи** и
роняет прогон, если данные всё-таки её перешагнули. Раньше «out-of-sample»
выводился двумя способами, и оба уезжали: бэктест пересчитывал индексы
walk-forward сплита (они сдвигаются при каждом росте CSV), а `data_fetch/holdout.py`
брал метку экспорта артефакта — точную, но она говорит лишь что модель уже
видела, и не мешает следующему прогону обучиться на том, что мы собирались
измерять.

Дата фиксированная, а не «последние N дней»: иначе каждый рефетч потихоньку
отдавал бы обучению кусок оценочного набора.

`meta.json` теперь несёт блок `provenance` (`train_end_ts`, `calibration_start_ts`,
`frozen_eval_start`, размеры спанов, embargo), поэтому оценке больше не нужно
догадываться: `data_fetch/holdout.py` берёт `train_end_ts`, если он есть, и метку
экспорта только для старых артефактов. Именно это позволяет оценивать модель,
собранную сегодня, на августе — экспорт 2026-09-07, а обучение закончилось
2026-07-31.

## Holdout: свечи, которых модели не видели

Все цифры выше посчитаны на `data/train/*.csv` — на том же ряде, на котором
модели обучались. Даже блок «out-of-sample» в бэктесте только **оценивал**
границу, пересчитывая walk-forward сплит, а эта оценка едет при каждом росте CSV.

Здесь догадки нет. Граница — **метка экспорта артефакта**: run id это timestamp
(`2026-08-17T15-47-17`), значит свеча с более поздним `ts` физически не могла
попасть в обучение — ONNX-файл уже лежал на диске, когда этот бар закрылся.
Никакого учёта в `meta.json` не требуется, и граница не может «поехать».

```bash
# скачать/обновить holdout-свечи (BTC/ETH/SOL + макро FRED)
python -m technical_analysis_trainer.data_fetch.holdout
python -m technical_analysis_trainer.data_fetch.holdout --status    # только отчёт по кэшу
python -m technical_analysis_trainer.data_fetch.holdout --refresh   # принудительно

# оценка обеих моделей на невиденных свечах
python -m technical_analysis_trainer.apps.holdout_eval
python -m technical_analysis_trainer.apps.holdout_eval --model trend --offline
```

Данные кэшируются в `data/holdout/` (git-ignored) и перекачиваются только когда
кэш отсутствует, старше `--max-age-hours` (12 по умолчанию) или не покрывает
cutoff плюс прогрев фич. Сеть на каждом прогоне тестов — это медленно, упирается
в рейт-лимит, падает офлайн и двигает метрики между двумя запусками одного и того
же кода. `--offline` (или `HOLDOUT_OFFLINE=1` для тестов) запрещает докачку.

Побочный эффект, который стоит отдельного упоминания: holdout тянет свежий SOL,
поэтому `xmkt_log_ret_sol` перестаёт быть константой — в обучающем CSV он
обрывается на 2026-04-17 и forward-fill'ится молча.

### Quantile на 495 невиденных барах (2026-08-17 → 2026-09-07)

| Метрика | Holdout | `meta.json` (test-фолд) |
|---|---|---|
| покрытие | **91.09%** при номинале 90% (CI 88.46–93.52%) | 91.34% |
| pinball skill vs uncond | +0.054 / +0.082 на хвостах, +0.003 на медиане | — |
| pinball skill vs HAR | −0.021 (q05) … +0.002 (q95) | +0.048 / +0.025 |
| MAE vs persistence | +0.0038 (CI −2.83…+4.48 — то есть ноль) | −0.0015 |

Калибровка интервалов держится вне обучающего периода — единственное, что здесь
демонстрируемо обобщается. Медиана по-прежнему не бьёт persistence.

### Trend на замороженном августе (2026-08-01 → 2026-09-06)

Прогон `2026-09-07T07-52-06`, обучение до 2026-03-12, калибровка до 2026-07-31,
оценка на 871 якоре (731 разрешённых, **эффективных 141** — метки перекрываются).

| Метрика | Значение |
|---|---|
| ROC-AUC | 0.5172 |
| accuracy | 46.65% (95% block CI 38.03–52.94%) |
| majority (всегда UP) | 60.74% |
| edge | −14.09 pp |
| mean p(UP) vs факт | 0.4914 vs 0.6074 (разрыв +0.116) |
| top-decile accuracy | 68.49% (n=73) |

Читать это надо так. **−14 pp — это почти целиком неверный порог, а не неверный
ранжир.** Порог, подобранный оракулом прямо на этом окне (то есть верхняя
граница, недостижимая честно), даёт 62.65% — против 60.74% у «всегда UP». То
есть даже идеальная калибровка вытаскивает модель ровно до базлайна, **+1.92 pp**
при эффективном n=141. Ранжир не несёт информации: AUC 0.5172 при таком n
неотличим от 0.5.

Калибратор обучен на марте–июле 2026 (доля UP 51.9%), а в августе она 60.7% —
это сдвиг априорной вероятности, который Platt-поправка по построению не
отслеживает (п. 4.4 плана, сознательно отложен).

### Скользящая оценка: 36 месячных origin'ов

Одно окно на 141 эффективном наблюдении не отвечает ни на один вопрос — CI по
accuracy шириной ~25 pp. `apps/rolling_eval.py` обучает по модели на каждый
месяц и оценивает её на следующем:

```bash
PYTHONPATH=. python -m technical_analysis_trainer.apps.rolling_eval --months 36 --seeds 5
# те же результаты как интерактивный отчёт (4 панели Plotly), без переобучения
PYTHONPATH=. python -m technical_analysis_trainer.apps.rolling_report \
    --json artifacts/rolling_eval_fixed_noac.json \
    --compare artifacts/rolling_eval_fixed_ac.json
```

Результат по 36 origin'ам, 42 фичи, с исправленным выравниванием спанов
(до исправления — 0.5373 и 27/36 на 17162 якорях; см. plan.txt §16):

| Метрика | Значение |
|---|---|
| AUC, среднее по месяцам | 0.5512 (std 0.0688, от 0.4083 до 0.6891) |
| месяцев с AUC > 0.5 | **28/36**, односторонний знаковый тест **p = 0.0006** |
| pooled AUC | 0.5361 (n=23030) |

Эдж в ранжире есть, и он маленький. Виден только потому, что окон 36: разброс
месячных AUC от 0.41 до 0.69 означает, что любое одно окно — включая
замороженный август — не говорит ничего.

Для сравнения, на тех же якорях: тривиальный бейзлайн «скользящая база»
даёт per-month AUC 0.4217 (5/36 выше 0.5), модель обыгрывает его на +0.1295 AUC
в месяц (95% CI [+0.0972, +0.1619], 33/36 месяцев).

Accuracy по покрытию — сигнал концентрируется там, где модель уверена:

| покрытие | n | модель | база |
|---|---|---|---|
| 10% | 2303 | 55.84% | 52.80% |
| 20% | 4606 | 53.76% | 51.61% |
| 100% | 23030 | 51.38% | 51.12% |

### Prior shift: принят, затем отменён (2026-09-08)

Скользящая доля UP среди **уже разрешившихся** барьеров (`trailing_prior`,
метка засчитывается на бар своего разрешения, не якоря). Был принят парно по
18 месяцам: accuracy +4.48 pp (p = 0.037), Brier +0.0054 (p = 0.0004).

**Тот замер был недействителен.** `build_span` возвращал строки, а не бары
(`add_features` отбрасывает ~51% баров: неразрешённые метки, NaN прогрева), и
`rolling_eval` брал prior по индексу, который в среднем шёл на **+47 баров
впереди** настоящего якоря. Prior на 50 баров вперёд уже впитал исход той самой
метки, которую предсказывает:

| где читается prior | pooled AUC |
|---|---|
| на истинном баре якоря | 0.5066 |
| на +50 барах вперёд | 0.5827 |
| по старому индексу (+47 в среднем) | 0.5738 |

Пересчёт на 36 месяцах с исправленным выравниванием:

| | среднее | 95% CI | p | месяцев лучше |
|---|---|---|---|---|
| accuracy | +1.87 pp | [−1.03, +4.77] | 0.21 | 19/36 |
| Brier | −0.0013 | [−0.0043, +0.0017] | 0.40 | 11/36 |

Brier сменил знак. Сам по себе trailing prior даёт per-month AUC 0.4217 и выше
0.5 лишь в 5 из 36 месяцев — на этом горизонте база тянет решение не в ту
сторону. Выключен (`prior_shift: false`); флаг оставлен. Развёрнутый артефакт
2026-09-07T11-29-30 всё ещё несёт поправку в бандле — конфиг влияет на следующий
прогон обучения, не на текущий.

Выравнивание закреплено тестами `models/trend/test_spans.py`; 3 из 5 падают на
старом коде. Подробности — plan.txt, раздел 16.

### Order flow: измеримой пользы нет

Шесть фич из `taker_buy_base` (агрессивная сторона объёма). Парно по тем же
18 месяцам, 52 фичи против 58:

| | среднее | 95% CI | p | месяцев лучше |
|---|---|---|---|---|
| AUC | +0.0092 | [−0.0138, +0.0323] | 0.433 | 10/18 |
| accuracy | +0.0038 | [−0.0122, +0.0197] | 0.645 | 9/18 |
| Brier | +0.0005 | [−0.0018, +0.0028] | 0.675 | 7/18 |

Все точечные оценки положительные, ни одна не отличима от нуля; знаковый тест по
AUC при этом ухудшился (13/18 → 11/18). Фичи оставлены — данные бесплатны, это
единственная знаковая информация о потоке в модели — но **не доказаны**, и
отношение «фичи / эффективный размер выборки» они ухудшают (58 входов на ~4500
эффективных меток). Флаг `order_flow: false` выключает блок для перепроверки.

### Отбор фич: 58 → 42 без потерь

`apps/feature_importance.py` считает permutation importance на **val**, блоками
(коррелированные семейства вроде трёх окон amihud перестановкой по одной фиче
недооцениваются). Средняя потеря val AUC при перестановке блока, 3 фолда × 3 сида:

| блок | вклад |
|---|---|
| xmkt returns/spreads | **+0.0297** |
| mtf 1d | +0.0127 |
| funding | +0.0089 |
| taker order flow | +0.0075 |
| amihud / range_per_vol | +0.0056 / +0.0053 |
| *шумовой пол по сидам* | *0.005* |
| xmkt_corr_main_eth | +0.0017 |
| mtf 4h | +0.0012 |
| momentum (roc/macd) | −0.0008 |
| dist_from_high | −0.0039 |
| atr_ratio | −0.0042 |

19 блоков из 25 — ниже шумового пола. Кросс-рыночные доходности важнее всего
остального вдвое.

Выброшено 16 входов (`feature_exclude` в конфиге; колонки продолжают считаться,
режется только список входов). Проверка парно по тем же 18 месяцам против набора
из 58 фич:

| | среднее | 95% CI | p | месяцев лучше |
|---|---|---|---|---|
| AUC | −0.0013 | [−0.0229, +0.0204] | 0.910 | 7/18 |
| accuracy | −0.0018 | [−0.0135, +0.0100] | 0.770 | 9/18 |

Минус 28% входов не стоят ничего измеримого. Принято по соображению
экономности: меньше параметров на ~4500 эффективных меток, меньше входов,
способных уехать (в выброшенном как раз блок с дрейфом −0.87…−1.66 sd), и
13/18 месяцев с AUC > 0.5 вернулись (на 58 фичах было 11/18). Это **не
улучшение**, это то же качество за меньшее.

Наборы 52, 58 и 42 неотличимы друг от друга; все три дают AUC ≈ 0.53 и 11–13
месяцев из 18 выше 0.5.

**Набор входов задаёт артефакт, не конфиг.** Инференс собирает все доступные
колонки и выбирает по списку из бандла (`features_override`) — прод читает
`test_config.yaml`, обучение `config.yaml`, и расхождение между ними один раз уже
вылезло как «X has 58 features, but StandardScaler is expecting 42». Тот же
принцип, что и для `target_spec` и калибратора: модель — единственный источник
правды о себе.

### Честное сравнение со старым артефактом

На **одном и том же** окне (2026-08-17 → 2026-09-07, n=389, эффективных 76):

| | старый `2026-08-17T15-47-17` | новый `2026-09-07T07-52-06` |
|---|---|---|
| ROC-AUC | 0.4794 | 0.4558 |
| accuracy | 48.07% (block CI 35.22–61.44) | 44.22% (block CI 32.13–55.27) |
| edge vs majority | −13.11 pp | −16.97 pp |

Новая модель на общем окне **не лучше**, а точечно хуже. Разница между 0.5172
(окно с 1 августа) и 0.4794 (окно с 17 августа) — это эффект окна, а не
улучшение модели: сравнивать модели на разных окнах нельзя, и первый вариант
этого README чуть не сделал именно такой вывод. При block CI шириной ~25 pp обе
модели неотличимы ни друг от друга, ни от монетки.

Что реально изменилось — не метрика, а то, что её теперь видно: доверительные
интервалы учитывают перекрытие меток, эффективное n печатается рядом с обычным,
граница обучения зафиксирована в конфиге и проверяется, а выбор прогона идёт по
`auc_val`. Раньше всё это было настроено так, что подобный результат выглядел бы
как 0.5855.

## Бэктест: прогон trend-модели по истории

Проигрывает ту же модель, что отдаёт прод, по историческим свечам из локальных
CSV — без сети и без БД. В консоль печатает отчёт по метрикам, в
`artifacts/backtest.html` кладёт интерактивный график, где каждое предсказание —
линия от своей свечи до барьера, который модель назвала, длиной ровно в горизонт
(24 бара). Соседние линии красятся разными цветами (палитра из 24 цветов на
24-барный горизонт), так что пересекающиеся линии всегда различимы.

```bash
# последние 500 свечей (по умолчанию, см. apps/backtest_config.yaml)
python -m technical_analysis_trainer.apps.backtest
# произвольный диапазон
python -m technical_analysis_trainer.apps.backtest --start 2026-01-01 --end 2026-06-01
# вся история, только отчёт в консоль
python -m technical_analysis_trainer.apps.backtest --last-bars 0 --no-chart
```

Диапазон, шаг, путь к HTML и предел числа линий задаются в
[`technical_analysis_trainer/apps/backtest_config.yaml`](technical_analysis_trainer/apps/backtest_config.yaml);
любой ключ перебивается флагом CLI (`--help`). `trend_run: null` означает «тот же
прогон, что в проде» — берётся `TREND_MODEL_RUN`, иначе `defaults.trend_run` из
`assembly/config.yaml`.

В блоке `data:` можно перебить пути к бенчмаркам и макро (`eth`, `sol`, `dxy`,
`risk_proxy`) — меняются только **пути**, а окна и набор фич по-прежнему берутся
из trend-конфига, иначе модель считалась бы по входам, которых не видела при
обучении. Так `holdout_eval.py` подставляет свежие CSV. Ключ `unseen_from`
задаёт жёсткую границу out-of-sample; при `null` она оценивается пересчётом
walk-forward сплита и отмечается в отчёте как `estimated`.

Источник, который заканчивается раньше последнего якоря, отчёт называет
поимённо (`⚠ stale input`): `attach_cross_market_features` его forward-fill'ит,
так что фича молча замерзает, а модель продолжает по ней оцениваться.

Модель триплбарьерная, поэтому и оценивается как триплбарьерная: эталон считает
`features.triple_barrier_labels` — та же функция, что делала метки при обучении.
Якоря, у которых за `max_holding` не задет ни один барьер, в точность не входят
(как и при обучении) и показываются отдельной строкой `unresolved`.
Проверка: прогон по тестовому фолду 2 даёт ROC-AUC 0.5860 / accuracy 55.61%
против 0.5855 / 55.40%, которые напечатал `train_v2.py` для того же прогона.

## Backfill: заполнение пропусков в предсказаниях

Дашборд накладывает сохранённые прогнозы на свечи; свеча без строки в БД —
дырка на графике (планировщик лежал, контейнер перезапускался и т.п.).

```bash
# посчитать, но не писать
python -m technical_analysis_trainer.apps.backfill_predictions --bars 200 --dry-run
# заполнить пропуски
python -m technical_analysis_trainer.apps.backfill_predictions --bars 200
# пересчитать всё заново
python -m technical_analysis_trainer.apps.backfill_predictions --bars 200 --force
```

Через API: `POST /api/backfill?bars=200`, статус — `GET /api/backfill-status`.
Автозапуск при старте: `BACKFILL_ON_START=1` (по умолчанию выключен; выполняется
фоном и никогда не блокирует старт API).

### Почему здесь нет утечки

Наивный backfill скармливает модели всю историю, и прогноз «подглядывает» в
собственное будущее — точность в панели статистики становится завышенной.
Здесь гарантия структурная, а не на доверии:

1. каждый входной фрейм (BTC + кросс-рынок) обрезается до `ts <= T` **до**
   расчёта любой фичи — будущих строк просто не существует внутри вызова;
2. обрезка проверяется `assert`, а не предполагается;
3. окно якорится так, что его содержимое совпадает с live-путём в момент T
   (`drop_last` для quantile, `last_row_is_closed` для trend).

Проверка: `python -m technical_analysis_trainer.apps.test_backfill_predictions`
— в частности, прогноз для T обязан быть **побитово одинаковым** с фреймом,
содержащим будущие свечи, и без них. Тест гоняется по holdout-свечам (см. выше):
на данных, которые модель запомнила, окно-баг может спрятаться — два слегка
разных входа дадут один и тот же ответ. `test_anchors_are_unseen` проверяет саму
предпосылку, а не полагается на неё.

## Prediction dashboard (local)

Backend serves JSON (`candles`, live quantile forecast for the next hour, trend probabilities, and optional `quantile_history` from PostgreSQL). When `DATABASE_URL` is set and the scheduler has written rows, `/api/snapshot` includes past quantile forecasts for the visible candle window; the chart draws them as faded dashed segments. The trend pipeline needs the predict CSV from [`technical_analysis_trainer/models/trend/test_config.yaml`](technical_analysis_trainer/models/trend/test_config.yaml) (`data_path`) and ONNX/scaler under `models/trend/models/`. Quantile artifacts live under `technical_analysis_trainer/models/quantile/models/{QUANTILE_MODEL_RUN}/` (env `QUANTILE_MODEL_RUN`).

From the **repository root** (`tan-model/`):

1. API (port 8001):

```bash
PYTHONPATH=. uvicorn technical_analysis_trainer.apps.api:app --reload --port 8001
```

2. Frontend (Vite proxies `/api` to 8001):

```bash
cd web && npm install && npm run dev
```

Open `http://localhost:5173`. Use **Refresh** to reload `/api/snapshot`.

## Docker full stack (postgres + api + web + scheduler)

Runs the dashboard behind nginx on port 80, FastAPI backend, PostgreSQL, an hourly prediction scheduler (supercronic), and the Go trading bot (paper mode by default).

```bash
cp deploy/.env.example deploy/.env
# Edit deploy/.env (POSTGRES_PASSWORD, model run ids)

# Ensure predict CSV exists (not in git):
# technical_analysis_trainer/data/predict/btcusdt_1h_predict.csv

docker compose -f deploy/docker-compose.yml --env-file deploy/.env up --build -d
```

- Dashboard: `http://localhost/` (or `WEB_PORT` from `.env`)
- Health: `curl http://localhost/health`
- API (internal): `http://api:8000/api/snapshot`

Stop: `docker compose -f deploy/docker-compose.yml --env-file deploy/.env down`

## Trading bot (Go)

The `trading_bot/` service polls predictions from the API (or PostgreSQL), evaluates a YAML strategy, and executes trades in **paper** or **live** mode.

Config: [`trading_bot/config/strategy.yaml`](trading_bot/config/strategy.yaml) (see [`strategy.example.yaml`](trading_bot/config/strategy.example.yaml)).

| Env var | Purpose |
|---------|---------|
| `STRATEGY_CONFIG` | Path to strategy YAML (default `config/strategy.yaml`) |
| `STATE_PATH` | JSON state file for position + dedupe across restarts (default `/var/lib/trading_bot/state.json`) |
| `TRADE_LOG_PATH` | Append structured trade logs (JSON lines) |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Required only for `bot.mode: live` |

**SL/TP behavior:**
- **live** — after market entry, places Binance protective orders (`STOP_LOSS` + `TAKE_PROFIT` on spot; `STOP_MARKET` + `TAKE_PROFIT_MARKET` on futures). Cancels them on manual close. `SyncPosition` detects exchange-side fills.
- **paper** — checks SL/TP against the last candle close on each cron tick.

Local run (paper, against API on port 8001):

```bash
cd trading_bot
STRATEGY_CONFIG=config/strategy.yaml STATE_PATH=/tmp/trading_bot_state.json \
  go run ./cmd/bot
```

Tests: `cd trading_bot && go test ./...`

Bot health: `curl http://localhost:8081/health` (exposed in Docker on the internal network).

## Deploy to AWS EC2

See [deploy/aws/README.md](deploy/aws/README.md) for Terraform (`deploy/aws/terraform/`) or manual bootstrap (`deploy/aws/scripts/bootstrap-ec2.sh`).

GitHub Actions:

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — build web + Docker images on PR/push
- [`.github/workflows/deploy-ec2.yml`](.github/workflows/deploy-ec2.yml) — push to GHCR and deploy to EC2 on `main`

Secrets: [deploy/aws/github-secrets.md](deploy/aws/github-secrets.md). On EC2 use `docker-compose.prod.yml` with `TAN_API_IMAGE` / `TAN_WEB_IMAGE` pointing at GHCR.

## Scheduled predictions (cron + PostgreSQL)

Hourly quantile and trend predictions are persisted to PostgreSQL by the scheduler. The dashboard computes the **current** forecast on demand and overlays **historical** quantile segments from the DB when `DATABASE_URL` is available to the API (Docker stack by default).

### Setup

1. Install deps (includes `psycopg2-binary`):

```bash
pip install -r technical_analysis_trainer/requirements.txt
```

2. Start Postgres only (optional; full stack includes postgres):

```bash
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d postgres
```

Or use the full Docker stack above (scheduler writes predictions hourly).

3. Configure env (copy [`.env.predict.example`](.env.predict.example) → `.env.predict`, or use `deploy/.env`):

```bash
cp .env.predict.example .env.predict
export DATABASE_URL=postgresql://tan:tan@localhost:5432/tan_predictions
```

4. Create tables (either Peewee or SQL):

```bash
# Peewee
PYTHONPATH=. python -c "from technical_analysis_trainer.storage.prediction_store import init_db; init_db()"

# Or SQL
psql "$DATABASE_URL" -f technical_analysis_trainer/sql/migrations/001_predictions.sql
```

5. Run once manually:

```bash
set -a && source .env.predict && set +a
PYTHONPATH=. python -m technical_analysis_trainer.apps.scheduler
```

Exit code `0` prints `scheduled_prediction OK {...}`; non-zero on failure (suitable for cron alerts).

**Requirements:** same ONNX/scaler/CSV paths as the live dashboard (`QUANTILE_MODEL_RUN`, `models/trend/test_config.yaml` `data_path`, `models/trend/models/`).

### Cron (1h candles, UTC)

Run **5 minutes after each hour** so Binance has closed the previous bar:

```cron
5 * * * * cd /path/to/tan-model && set -a && . .env.predict && set +a && PYTHONPATH=. /path/to/venv/bin/python -m technical_analysis_trainer.apps.scheduler >> logs/scheduled_predict.log 2>&1
```

On macOS you can use `launchd` with the same `ProgramArguments` instead of crontab.

### Verify

```bash
psql "$DATABASE_URL" -c "SELECT target_ts, q50, created_at FROM quantile_predictions ORDER BY target_ts DESC LIMIT 5;"
psql "$DATABASE_URL" -c "SELECT target_ts, predicted_trend, prob_up, created_at FROM trend_predictions ORDER BY target_ts DESC LIMIT 5;"
```

Re-running the job for the same `target_ts` updates the row (UPSERT), it does not duplicate.