import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  BarChart3,
  Brain,
  CandlestickChart,
  GitBranch,
  LineChart as LineIcon,
  RefreshCw,
  Table2,
  TrendingUp,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  getBacktest,
  getEquity,
  getFeatureImportance,
  getModelComparison,
  getPdtTree,
  getPredictions,
  getSummary,
} from "./api";

const MODELS = [
  "PDT_direction_median_return",
  "XGBoostRegressor",
  "LSTMRegressor",
  "LastDailyReturn",
  "LastHorizonReturn",
  "NaivePrice",
];

const modelNames = {
  PDT_direction_median_return: "PDT",
  XGBoostRegressor: "XGBoost",
  LSTMRegressor: "LSTM",
  LastDailyReturn: "Last return",
  LastHorizonReturn: "Last horizon",
  NaivePrice: "Naive",
};

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function modelLabel(model) {
  return modelNames[model] || model;
}

function MetricCard({ icon, label, value, detail }) {
  return (
    <section className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div>
        <div className="metric-label">{label}</div>
        <div className="metric-value">{value}</div>
        <div className="metric-detail">{detail}</div>
      </div>
    </section>
  );
}

function Segmented({ items, value, onChange }) {
  return (
    <div className="segmented">
      {items.map((item) => (
        <button
          className={item.value === value ? "active" : ""}
          key={item.value}
          onClick={() => onChange(item.value)}
          type="button"
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

function Select({ label, value, onChange, options }) {
  return (
    <label className="select-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {modelLabel(option)}
          </option>
        ))}
      </select>
    </label>
  );
}

function StatTable({ rows, columns }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.model}-${row.secid}-${row.horizon}-${index}`}>
              {columns.map((column) => (
                <td key={column.key}>{column.render ? column.render(row) : row[column.key]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LoadingBlock() {
  return (
    <div className="loading">
      <RefreshCw size={18} />
      <span>загрузка данных</span>
    </div>
  );
}

function ChartArea({ children, large = false }) {
  const ref = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useLayoutEffect(() => {
    if (!ref.current) return undefined;

    const updateSize = () => {
      const rect = ref.current.getBoundingClientRect();
      setSize({
        width: Math.round(rect.width),
        height: Math.round(rect.height),
      });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div className={large ? "chart-box large" : "chart-box"} ref={ref}>
      {size.width > 20 && size.height > 20 ? children(size.width, size.height) : <LoadingBlock />}
    </div>
  );
}

export default function App() {
  const [secid, setSecid] = useState("RTSI");
  const [horizon, setHorizon] = useState(1);
  const [model, setModel] = useState("PDT_direction_median_return");
  const [tab, setTab] = useState("overview");
  const [summary, setSummary] = useState(null);
  const [comparison, setComparison] = useState([]);
  const [backtest, setBacktest] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [equity, setEquity] = useState([]);
  const [pdtFeatures, setPdtFeatures] = useState([]);
  const [xgbFeatures, setXgbFeatures] = useState([]);
  const [tree, setTree] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    getSummary().then(setSummary).catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    setError("");
    Promise.all([
      getModelComparison(horizon, secid),
      getBacktest(horizon, secid),
      getPredictions(horizon, secid, model),
      getEquity(horizon, secid, model),
      getFeatureImportance("pdt", horizon, secid),
      getFeatureImportance("xgboost", horizon, secid),
      getPdtTree(horizon, secid),
    ])
      .then(([comparisonRows, backtestRows, predictionRows, equityRows, pdtRows, xgbRows, treeResult]) => {
        setComparison(comparisonRows);
        setBacktest(backtestRows);
        setPredictions(predictionRows);
        setEquity(equityRows);
        setPdtFeatures(pdtRows);
        setXgbFeatures(xgbRows);
        setTree(treeResult.text);
      })
      .catch((err) => setError(err.message));
  }, [horizon, secid, model]);

  const availableModels = useMemo(() => {
    const names = comparison.map((row) => row.model);
    return MODELS.filter((name) => names.includes(name));
  }, [comparison]);

  useEffect(() => {
    if (availableModels.length > 0 && !availableModels.includes(model)) {
      setModel(availableModels[0]);
    }
  }, [availableModels, model]);

  const currentModelRow = comparison.find((row) => row.model === model);
  const currentBacktestRow = backtest.find((row) => row.model === model);

  const predictionChart = predictions.map((row) => ({
    date: row.future_date,
    actual: row.future_close,
    forecast: row.predicted_close,
  }));

  const equityChart = equity.map((row) => ({
    date: row.future_date,
    strategy: row.strategy_equity,
    buyHold: row.buy_hold_equity,
  }));

  const accuracyChart = comparison.map((row) => ({
    model: modelLabel(row.model),
    balanced: Number(row.balanced_accuracy || 0),
    accuracy: Number(row.accuracy || 0),
  }));

  const rmseChart = comparison.map((row) => ({
    model: modelLabel(row.model),
    rmse: Number(row.rmse || 0),
  }));

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">курсовая работа · PDT для MOEX / RTSI</p>
          <h1>Панель сравнения моделей</h1>
        </div>
        <div className="control-row">
          <Segmented
            items={[
              { label: "RTSI", value: "RTSI" },
              { label: "IMOEX", value: "IMOEX" },
            ]}
            value={secid}
            onChange={setSecid}
          />
          <Segmented
            items={[
              { label: "h=1", value: 1 },
              { label: "h=5", value: 5 },
            ]}
            value={horizon}
            onChange={setHorizon}
          />
          <Select label="модель" value={model} onChange={setModel} options={availableModels.length ? availableModels : MODELS} />
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      {!summary ? (
        <LoadingBlock />
      ) : (
        <>
          <section className="metrics-grid">
            <MetricCard
              icon={<Brain size={22} />}
              label="PDT, RTSI h=1"
              value={formatPercent(summary.pdt_rtsi_h1.balanced_accuracy)}
              detail={`balanced accuracy, test ${summary.pdt_rtsi_h1.test_first_date} — ${summary.pdt_rtsi_h1.test_last_date}`}
            />
            <MetricCard
              icon={<Activity size={22} />}
              label="лучшее направление"
              value={modelLabel(summary.best_accuracy.model)}
              detail={`${summary.best_accuracy.secid}, h=${summary.best_accuracy.horizon}, BA ${formatPercent(summary.best_accuracy.balanced_accuracy)}`}
            />
            <MetricCard
              icon={<BarChart3 size={22} />}
              label="минимальный RMSE"
              value={modelLabel(summary.best_rmse.model)}
              detail={`${summary.best_rmse.secid}, h=${summary.best_rmse.horizon}, RMSE ${formatNumber(summary.best_rmse.rmse, 2)}`}
            />
            <MetricCard
              icon={<TrendingUp size={22} />}
              label="лучший backtest"
              value={formatPercent(summary.best_return.strategy_total_return)}
              detail={`${modelLabel(summary.best_return.model)}, ${summary.best_return.secid}, h=${summary.best_return.horizon}`}
            />
          </section>

          <nav className="tabs">
            <button className={tab === "overview" ? "active" : ""} onClick={() => setTab("overview")} type="button">
              <Table2 size={17} />
              сводка
            </button>
            <button className={tab === "predictions" ? "active" : ""} onClick={() => setTab("predictions")} type="button">
              <LineIcon size={17} />
              прогноз
            </button>
            <button className={tab === "backtest" ? "active" : ""} onClick={() => setTab("backtest")} type="button">
              <CandlestickChart size={17} />
              стратегия
            </button>
            <button className={tab === "tree" ? "active" : ""} onClick={() => setTab("tree")} type="button">
              <GitBranch size={17} />
              PDT
            </button>
          </nav>

          {tab === "overview" && (
            <section className="view-grid">
              <div className="panel">
                <div className="panel-head">
                  <h2>Качество классификации направления</h2>
                  <span>{secid}, h={horizon}</span>
                </div>
                <ChartArea>
                  {accuracyChart.length ? (
                    (width, height) => (
                      <BarChart data={accuracyChart} width={width} height={height}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="model" tick={{ fontSize: 12 }} />
                        <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(value * 100)}%`} />
                        <Tooltip formatter={(value) => formatPercent(value, 2)} />
                        <Legend />
                        <Bar dataKey="balanced" name="balanced accuracy" fill="#2f6f73" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="accuracy" name="accuracy" fill="#a45f38" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    )
                  ) : (
                    () => <LoadingBlock />
                  )}
                </ChartArea>
              </div>

              <div className="panel">
                <div className="panel-head">
                  <h2>Ошибка прогноза цены</h2>
                  <span>RMSE</span>
                </div>
                <ChartArea>
                  {rmseChart.length ? (
                    (width, height) => (
                      <BarChart data={rmseChart} width={width} height={height}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="model" tick={{ fontSize: 12 }} />
                        <YAxis />
                        <Tooltip formatter={(value) => formatNumber(value, 2)} />
                        <Bar dataKey="rmse" name="RMSE" fill="#415a77" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    )
                  ) : (
                    () => <LoadingBlock />
                  )}
                </ChartArea>
              </div>

              <div className="panel wide">
                <div className="panel-head">
                  <h2>Таблица моделей</h2>
                  <span>test-period</span>
                </div>
                <StatTable
                  rows={comparison}
                  columns={[
                    { key: "model", label: "модель", render: (row) => modelLabel(row.model) },
                    { key: "mae", label: "MAE", render: (row) => formatNumber(row.mae, 2) },
                    { key: "rmse", label: "RMSE", render: (row) => formatNumber(row.rmse, 2) },
                    { key: "mape_pct", label: "MAPE", render: (row) => `${formatNumber(row.mape_pct, 2)}%` },
                    { key: "accuracy", label: "accuracy", render: (row) => formatPercent(row.accuracy, 1) },
                    { key: "balanced_accuracy", label: "balanced acc.", render: (row) => formatPercent(row.balanced_accuracy, 1) },
                    { key: "predicted_positive_share", label: "long share", render: (row) => formatPercent(row.predicted_positive_share, 1) },
                  ]}
                />
              </div>
            </section>
          )}

          {tab === "predictions" && (
            <section className="view-grid">
              <div className="panel wide">
                <div className="panel-head">
                  <h2>Факт и прогноз</h2>
                  <span>{modelLabel(model)}, {secid}, h={horizon}</span>
                </div>
                <ChartArea large>
                  {predictionChart.length ? (
                    (width, height) => (
                      <LineChart data={predictionChart} width={width} height={height}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="date" minTickGap={36} tick={{ fontSize: 12 }} />
                        <YAxis domain={["auto", "auto"]} />
                        <Tooltip formatter={(value) => formatNumber(value, 2)} />
                        <Legend />
                        <Line dataKey="actual" name="future close" stroke="#2f6f73" strokeWidth={2} dot={false} />
                        <Line dataKey="forecast" name="predicted close" stroke="#a45f38" strokeWidth={2} dot={false} />
                      </LineChart>
                    )
                  ) : (
                    () => <LoadingBlock />
                  )}
                </ChartArea>
              </div>
              <div className="panel">
                <div className="panel-head">
                  <h2>Текущая модель</h2>
                  <span>test</span>
                </div>
                {currentModelRow && (
                  <div className="model-stats">
                    <div><span>MAE</span><strong>{formatNumber(currentModelRow.mae, 2)}</strong></div>
                    <div><span>RMSE</span><strong>{formatNumber(currentModelRow.rmse, 2)}</strong></div>
                    <div><span>MAPE</span><strong>{formatNumber(currentModelRow.mape_pct, 2)}%</strong></div>
                    <div><span>balanced accuracy</span><strong>{formatPercent(currentModelRow.balanced_accuracy)}</strong></div>
                    <div><span>precision up</span><strong>{formatPercent(currentModelRow.precision_up)}</strong></div>
                    <div><span>recall up</span><strong>{formatPercent(currentModelRow.recall_up)}</strong></div>
                  </div>
                )}
              </div>
            </section>
          )}

          {tab === "backtest" && (
            <section className="view-grid">
              <div className="panel wide">
                <div className="panel-head">
                  <h2>Equity curve</h2>
                  <span>{modelLabel(model)} против buy-and-hold</span>
                </div>
                <ChartArea large>
                  {equityChart.length ? (
                    (width, height) => (
                      <LineChart data={equityChart} width={width} height={height}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="date" minTickGap={36} tick={{ fontSize: 12 }} />
                        <YAxis domain={["auto", "auto"]} />
                        <Tooltip formatter={(value) => formatNumber(value, 3)} />
                        <Legend />
                        <Line dataKey="strategy" name="strategy" stroke="#2f6f73" strokeWidth={2} dot={false} />
                        <Line dataKey="buyHold" name="buy-and-hold" stroke="#415a77" strokeWidth={2} dot={false} />
                      </LineChart>
                    )
                  ) : (
                    () => <LoadingBlock />
                  )}
                </ChartArea>
              </div>
              <div className="panel">
                <div className="panel-head">
                  <h2>Показатели стратегии</h2>
                  <span>cost 5 bps</span>
                </div>
                {currentBacktestRow && (
                  <div className="model-stats">
                    <div><span>total return</span><strong>{formatPercent(currentBacktestRow.strategy_total_return)}</strong></div>
                    <div><span>buy hold</span><strong>{formatPercent(currentBacktestRow.buy_hold_total_return)}</strong></div>
                    <div><span>sharpe</span><strong>{formatNumber(currentBacktestRow.strategy_sharpe, 2)}</strong></div>
                    <div><span>max drawdown</span><strong>{formatPercent(currentBacktestRow.strategy_max_drawdown)}</strong></div>
                    <div><span>exposure</span><strong>{formatPercent(currentBacktestRow.exposure)}</strong></div>
                    <div><span>entries</span><strong>{currentBacktestRow.entries}</strong></div>
                  </div>
                )}
              </div>
              <div className="panel wide">
                <div className="panel-head">
                  <h2>Backtest по всем моделям</h2>
                  <span>{secid}, h={horizon}</span>
                </div>
                <StatTable
                  rows={backtest}
                  columns={[
                    { key: "model", label: "модель", render: (row) => modelLabel(row.model) },
                    { key: "strategy_total_return", label: "strategy", render: (row) => formatPercent(row.strategy_total_return, 1) },
                    { key: "buy_hold_total_return", label: "buy hold", render: (row) => formatPercent(row.buy_hold_total_return, 1) },
                    { key: "strategy_sharpe", label: "sharpe", render: (row) => formatNumber(row.strategy_sharpe, 2) },
                    { key: "strategy_max_drawdown", label: "max dd", render: (row) => formatPercent(row.strategy_max_drawdown, 1) },
                    { key: "exposure", label: "exposure", render: (row) => formatPercent(row.exposure, 1) },
                    { key: "entries", label: "entries" },
                  ]}
                />
              </div>
            </section>
          )}

          {tab === "tree" && (
            <section className="view-grid">
              <div className="panel">
                <div className="panel-head">
                  <h2>PDT feature importance</h2>
                  <span>{secid}, h={horizon}</span>
                </div>
                <ChartArea>
                  {pdtFeatures.length ? (
                    (width, height) => (
                      <BarChart data={pdtFeatures} layout="vertical" margin={{ left: 70 }} width={width} height={height}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="feature" tick={{ fontSize: 12 }} />
                        <Tooltip formatter={(value) => formatNumber(value, 3)} />
                        <Bar dataKey="importance" name="importance" fill="#2f6f73" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    )
                  ) : (
                    () => <LoadingBlock />
                  )}
                </ChartArea>
              </div>
              <div className="panel">
                <div className="panel-head">
                  <h2>XGBoost feature importance</h2>
                  <span>{secid}, h={horizon}</span>
                </div>
                <ChartArea>
                  {xgbFeatures.length ? (
                    (width, height) => (
                      <BarChart data={xgbFeatures} layout="vertical" margin={{ left: 70 }} width={width} height={height}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="feature" tick={{ fontSize: 12 }} />
                        <Tooltip formatter={(value) => formatNumber(value, 3)} />
                        <Bar dataKey="importance" name="importance" fill="#a45f38" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    )
                  ) : (
                    () => <LoadingBlock />
                  )}
                </ChartArea>
              </div>
              <div className="panel wide">
                <div className="panel-head">
                  <h2>Текст дерева PDT</h2>
                  <span>выгрузка из этапа 3</span>
                </div>
                <pre className="tree-box">{tree}</pre>
              </div>
            </section>
          )}
        </>
      )}
    </main>
  );
}
