import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from "react";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable
} from "@tanstack/react-table";

const EMPTY_LIST = [];
const TOOLTIP_LENGTH = 40;
const WRAPPED_COLUMNS = new Set(["event_group_id", "event_urls"]);
const DETAIL_ROUTE = /^\/event-groups\/([^/]+)$/;
const OPPORTUNITIES_ROUTE = "/open-markets";

export default function App() {
  const [rows, setRows] = useState(EMPTY_LIST);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");
  const [sorting, setSorting] = useState([]);
  const [activeOnly, setActiveOnly] = useState(false);
  const [path, setPath] = useState(window.location.pathname);
  const [detail, setDetail] = useState(null);
  const [detailStatus, setDetailStatus] = useState("idle");
  const [detailError, setDetailError] = useState("");
  const [opportunities, setOpportunities] = useState(EMPTY_LIST);
  const [opportunityStatus, setOpportunityStatus] = useState("idle");
  const [opportunityError, setOpportunityError] = useState("");
  const tableWrapRef = useRef(null);
  const tableScrollTopRef = useRef(null);
  const tableScrollTopInnerRef = useRef(null);
  const latestFutureTimelines = useMemo(() => {
    if (!detail || detail.future_timelines.length === 0) {
      return EMPTY_LIST;
    }
    const sorted = [...detail.future_timelines].sort(
      (a, b) => new Date(b.generated_at) - new Date(a.generated_at)
    );
    const latestGeneratedAt = sorted[0].generated_at;
    return sorted.filter(
      (timeline) => timeline.generated_at === latestGeneratedAt
    );
  }, [detail]);

  const route = useMemo(() => {
    const match = path.match(DETAIL_ROUTE);
    if (!match) {
      if (path === OPPORTUNITIES_ROUTE) {
        return { type: "opportunities" };
      }
      return { type: "table" };
    }
    return { type: "detail", id: match[1] };
  }, [path]);

  const navigate = useCallback((nextPath) => {
    if (window.location.pathname === nextPath) {
      return;
    }
    window.history.pushState({}, "", nextPath);
    setPath(nextPath);
  }, []);

  useEffect(() => {
    const onPopState = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  useEffect(() => {
    let alive = true;

    fetch("/event_stats_table")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Request failed: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (!alive) {
          return;
        }
        setRows(data);
        setStatus("ready");
      })
      .catch((fetchError) => {
        if (!alive) {
          return;
        }
        setError(fetchError.message);
        setStatus("error");
      });

    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    let alive = true;
    setOpportunityStatus("loading");
    setOpportunityError("");

    fetch("/open_market_opportunities")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Request failed: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (!alive) {
          return;
        }
        setOpportunities(data);
        setOpportunityStatus("ready");
      })
      .catch((fetchError) => {
        if (!alive) {
          return;
        }
        setOpportunityError(fetchError.message);
        setOpportunityStatus("error");
      });

    return () => {
      alive = false;
    };
  }, []);

  const { columns, numericColumns } = useMemo(() => {
    if (rows.length === 0 || route.type !== "table") {
      return { columns: [], numericColumns: new Set() };
    }

    const keys = Object.keys(rows[0]);
    const numericColumns = new Set(
      keys.filter((key) => rows.some((row) => typeof row[key] === "number"))
    );
    const focusKeys = ["event_names", "event_urls"];
    const baseKeys = keys.filter(
      (key) => !focusKeys.includes(key) && key !== "active"
    );
    const orderedKeys = [...baseKeys];

    if (keys.includes("active")) {
      const idIndex = orderedKeys.indexOf("event_group_id");
      if (idIndex === -1) {
        orderedKeys.unshift("active");
      } else {
        orderedKeys.splice(idIndex + 1, 0, "active");
      }
    }

    if (keys.includes("event_names")) {
      orderedKeys.splice(1, 0, "event_names");
    }

    if (keys.includes("event_urls")) {
      orderedKeys.splice(2, 0, "event_urls");
    }

    return {
      columns: orderedKeys.map((key) => ({
        accessorKey: key,
        header: key,
        cell: (info) => formatCellValue(info.column.id, info.getValue()),
        enableSorting: true
      })),
      numericColumns
    };
  }, [rows, route.type]);

  const visibleRows = useMemo(() => {
    if (route.type !== "table") {
      return EMPTY_LIST;
    }
    if (!activeOnly) {
      return rows;
    }
    return rows.filter((row) => row.active);
  }, [activeOnly, rows, route.type]);

  const table = useReactTable({
    data: visibleRows,
    columns,
    state: {
      sorting
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel()
  });

  useLayoutEffect(() => {
    const wrap = tableWrapRef.current;
    const top = tableScrollTopRef.current;
    const inner = tableScrollTopInnerRef.current;
    if (!wrap || !top || !inner) {
      return undefined;
    }
    const update = () => {
      inner.style.width = `${wrap.scrollWidth}px`;
      top.scrollLeft = wrap.scrollLeft;
    };
    update();
    window.addEventListener("resize", update);
    let resizeObserver;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(update);
      resizeObserver.observe(wrap);
    }
    return () => {
      window.removeEventListener("resize", update);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [activeOnly, route.type, sorting, status, visibleRows.length]);

  useEffect(() => {
    const wrap = tableWrapRef.current;
    const top = tableScrollTopRef.current;
    if (!wrap || !top) {
      return undefined;
    }
    let syncingFromTop = false;
    let syncingFromWrap = false;
    const onTopScroll = () => {
      if (syncingFromWrap) {
        syncingFromWrap = false;
        return;
      }
      syncingFromTop = true;
      wrap.scrollLeft = top.scrollLeft;
    };
    const onWrapScroll = () => {
      if (syncingFromTop) {
        syncingFromTop = false;
        return;
      }
      syncingFromWrap = true;
      top.scrollLeft = wrap.scrollLeft;
    };
    top.addEventListener("scroll", onTopScroll);
    wrap.addEventListener("scroll", onWrapScroll);
    return () => {
      top.removeEventListener("scroll", onTopScroll);
      wrap.removeEventListener("scroll", onWrapScroll);
    };
  }, [route.type, status, visibleRows.length]);

  function formatCellValue(key, value) {
    if (value === null || value === undefined) {
      return "";
    }

    if (key === "event_group_id") {
      const id = String(value);
      return (
        <a
          className="event-group-link"
          href={`/event-groups/${id}`}
          onClick={(event) => {
            event.preventDefault();
            navigate(`/event-groups/${id}`);
          }}
        >
          {id}
        </a>
      );
    }

    if (key === "active") {
      return (
        <span className={`status-badge ${value ? "is-active" : "is-inactive"}`}>
          {value ? "Active" : "Inactive"}
        </span>
      );
    }

    if (key === "event_urls") {
      const urls = String(value)
        .split("\n")
        .map((url) => url.trim())
        .filter((url) => url.length > 0);

      return (
        <div className="event-url-list">
          {urls.map((url, index) => (
            <a
              key={`${url}-${index}`}
              href={url}
              target="_blank"
              rel="noreferrer"
            >
              {url}
            </a>
          ))}
        </div>
      );
    }

    if (typeof value === "number") {
      return value.toFixed(2);
    }

    if (typeof value === "object") {
      return JSON.stringify(value);
    }

    const text = String(value);

    if (WRAPPED_COLUMNS.has(key)) {
      return text;
    }

    if (text.length > TOOLTIP_LENGTH) {
      return (
        <span className="cell-tooltip" data-tooltip={text}>
          <span className="cell-clip">{text}</span>
        </span>
      );
    }

    return <span className="cell-clip">{text}</span>;
  }

  function formatProbability(value) {
    if (value === null || value === undefined) {
      return "—";
    }
    return `${(value * 100).toFixed(1)}%`;
  }

  function getYesProbability(market) {
    if (!market?.outcomes || !market?.outcome_prices) {
      return null;
    }
    const yesIndex = market.outcomes.indexOf("Yes");
    if (yesIndex === -1) {
      return null;
    }
    return market.outcome_prices[yesIndex];
  }

  useEffect(() => {
    if (route.type !== "detail") {
      return;
    }
    let alive = true;
    setDetailStatus("loading");
    setDetailError("");
    setDetail(null);

    fetch(`/event_groups/${route.id}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Request failed: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (!alive) {
          return;
        }
        setDetail(data);
        setDetailStatus("ready");
      })
      .catch((fetchError) => {
        if (!alive) {
          return;
        }
        setDetailError(fetchError.message);
        setDetailStatus("error");
      });

    return () => {
      alive = false;
    };
  }, [route]);

  return (
    <div className="page">
      <header className="forecast-bar">
        <div className="forecast-brand">
          <div className="brand-badge">NT</div>
          <div>
            <p className="brand-title">WORLD FORECASTING</p>
            <p className="brand-subtitle">
              {route.type === "detail"
                ? "Event group profile"
                : route.type === "opportunities"
                  ? "Open market opportunities"
                  : "Event stats feed"}
            </p>
          </div>
        </div>
        <div className="forecast-status">
          <button
            className={`status-pill ${route.type === "table" ? "is-live" : ""}`}
            type="button"
            onClick={() => navigate("/")}
          >
            Event stats
          </button>
          <button
            className={`status-pill ${
              route.type === "opportunities" ? "is-live" : ""
            }`}
            type="button"
            onClick={() => navigate(OPPORTUNITIES_ROUTE)}
          >
            Opportunities
          </button>
        </div>
        <div className="forecast-ticker">
          {route.type === "opportunities" ? (
            <span>
              {opportunityStatus === "ready"
                ? `${opportunities.length} opportunities`
                : opportunityStatus === "loading"
                  ? "Loading opportunities"
                  : "Opportunity feed error"}
            </span>
          ) : route.type === "table" ? (
            <span>
              {status === "ready"
                ? `${rows.length} event groups`
                : status === "loading"
                  ? "Loading event stats"
                  : "Event stats error"}
            </span>
          ) : (
            <span>Event group detail</span>
          )}
        </div>
      </header>

      {route.type === "table" ? (
        <section className="panel">
          {status === "loading" && (
            <div className="state">Loading event stats table…</div>
          )}
          {status === "error" && (
            <div className="state error">{error}</div>
          )}
          {status === "ready" && rows.length === 0 && (
            <div className="state">No rows returned.</div>
          )}
          {status === "ready" && rows.length > 0 && (
            <>
              <div className="table-controls">
                <label className="toggle">
                  <input
                    type="checkbox"
                    checked={activeOnly}
                    onChange={(event) => setActiveOnly(event.target.checked)}
                  />
                  <span>Show active only</span>
                </label>
                <div className="table-meta">
                  Showing {visibleRows.length} of {rows.length} groups
                </div>
              </div>
              <div className="table-scroll-top" ref={tableScrollTopRef} aria-hidden="true">
                <div className="table-scroll-top-inner" ref={tableScrollTopInnerRef} />
              </div>
              <div className="table-wrap" ref={tableWrapRef}>
                <table>
                  <thead>
                    {table.getHeaderGroups().map((headerGroup) => (
                      <tr key={headerGroup.id}>
                        {headerGroup.headers.map((header) => (
                          <th
                            key={header.id}
                            data-col={header.column.id}
                            className={
                              numericColumns.has(header.column.id)
                                ? "numeric-col"
                                : undefined
                            }
                          >
                            {header.isPlaceholder ? null : (
                              <button
                                className="sort-button"
                                type="button"
                                onClick={header.column.getToggleSortingHandler()}
                              >
                                {flexRender(
                                  header.column.columnDef.header,
                                  header.getContext()
                                )}
                                <span className="sort-indicator">
                                  {header.column.getIsSorted() === "asc"
                                    ? "▲"
                                    : header.column.getIsSorted() === "desc"
                                      ? "▼"
                                      : "↕"}
                                </span>
                              </button>
                            )}
                          </th>
                        ))}
                      </tr>
                    ))}
                  </thead>
                  <tbody>
                    {table.getRowModel().rows.map((row) => (
                      <tr key={row.id}>
                        {row.getVisibleCells().map((cell) => (
                          <td
                            key={cell.id}
                            data-col={cell.column.id}
                            className={
                              numericColumns.has(cell.column.id)
                                ? "numeric-col"
                                : undefined
                            }
                          >
                            {flexRender(
                              cell.column.columnDef.cell,
                              cell.getContext()
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      ) : route.type === "opportunities" ? (
        <section className="panel">
          {opportunityStatus === "loading" && (
            <div className="state">Loading open market opportunities…</div>
          )}
          {opportunityStatus === "error" && (
            <div className="state error">{opportunityError}</div>
          )}
          {opportunityStatus === "ready" && opportunities.length === 0 && (
            <div className="state">No open market opportunities yet.</div>
          )}
          {opportunityStatus === "ready" && opportunities.length > 0 && (
            <div className="table-wrap">
              <table className="compact-table">
                <thead>
                  <tr>
                    <th>Event group</th>
                    <th>Open market</th>
                    <th className="numeric-col">Market probability</th>
                    <th className="numeric-col">Model probability</th>
                    <th className="numeric-col">OpenForecaster probability</th>
                    <th className="numeric-col">Simulations</th>
                    <th className="numeric-col">Deviation</th>
                  </tr>
                </thead>
                <tbody>
                  {opportunities.map((opportunity) => (
                    <tr key={opportunity.market_id}>
                      <td>
                        <a
                          className="event-group-link"
                          href={`/event-groups/${opportunity.event_group_id}`}
                          onClick={(event) => {
                            event.preventDefault();
                            navigate(`/event-groups/${opportunity.event_group_id}`);
                          }}
                        >
                          {opportunity.event_group_id}
                        </a>
                      </td>
                      <td>
                        <div className="opportunity-question">
                          <strong>{opportunity.market_question}</strong>
                          {opportunity.market_slug && (
                            <a
                              href={`https://polymarket.com/market/${opportunity.market_slug}`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              {opportunity.market_slug}
                            </a>
                          )}
                        </div>
                      </td>
                      <td className="numeric-col">
                        {formatProbability(opportunity.market_probability)}
                      </td>
                      <td className="numeric-col">
                        <span className="probability-badge">
                          {formatProbability(
                            opportunity.estimated_probability
                          )}
                        </span>
                      </td>
                      <td className="numeric-col">
                        <span className="probability-badge">
                          {formatProbability(
                            opportunity.openforecaster_probability
                          )}
                        </span>
                      </td>
                      <td className="numeric-col">{opportunity.samples}</td>
                      <td className="numeric-col">
                        <span className="delta-badge">
                          {formatProbability(opportunity.probability_delta)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      ) : (
        <section className="panel">
          <div className="detail-header">
            <button
              className="ghost-button"
              type="button"
              onClick={() => navigate("/")}
            >
              Back to stats table
            </button>
          </div>
          {detailStatus === "loading" && (
            <div className="state">Loading event group…</div>
          )}
          {detailStatus === "error" && (
            <div className="state error">{detailError}</div>
          )}
          {detailStatus === "ready" && detail && (
            <div className="detail-body">
              <div className="detail-title">
                <div>
                  <p className="detail-id">{detail.event_group_id}</p>
                  <h2>{detail.event_group_title}</h2>
                </div>
                <span
                  className={`status-badge ${
                    detail.active ? "is-active" : "is-inactive"
                  }`}
                >
                  {detail.active ? "Active" : "Inactive"}
                </span>
              </div>

              <div className="detail-section">
                <h3>Event group events</h3>
                <div className="event-grid">
                  {detail.events.map((event) => (
                    <article key={event.id} className="event-card">
                      <div className="event-header">
                        <h4>{event.title}</h4>
                        <a href={event.url} target="_blank" rel="noreferrer">
                          {event.url}
                        </a>
                      </div>
                      {event.description && (
                        <p className="event-description">{event.description}</p>
                      )}
                      <div className="event-meta">
                        <span>Start: {event.start_date || "—"}</span>
                        <span>End: {event.end_date || "—"}</span>
                      </div>
                    </article>
                  ))}
                </div>
              </div>

              <div className="detail-section">
                <h3>Open markets</h3>
                {detail.open_markets.length === 0 ? (
                  <div className="state">No open markets listed.</div>
                ) : (
                  <div className="table-wrap">
                    <table className="compact-table">
                      <thead>
                        <tr>
                          <th>Question</th>
                          <th>End date</th>
                          <th className="numeric-col">Liquidity</th>
                          <th className="numeric-col">Volume</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.open_markets.map((market) => (
                          <tr key={market.id}>
                            <td>
                              {market.slug ? (
                                <a
                                  href={`https://polymarket.com/market/${market.slug}`}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  {market.question}
                                </a>
                              ) : (
                                market.question
                              )}
                            </td>
                            <td>{market.end_date || "—"}</td>
                            <td className="numeric-col">
                              {market.liquidity !== null
                                ? market.liquidity.toFixed(2)
                                : "—"}
                            </td>
                            <td className="numeric-col">
                              {market.volume !== null
                                ? market.volume.toFixed(2)
                                : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <div className="detail-section">
                <h3>Market probabilities</h3>
                {detail.open_markets.length === 0 ? (
                  <div className="state">No open markets listed.</div>
                ) : (
                  <div className="table-wrap">
                    <table className="compact-table probability-table">
                      <thead>
                        <tr>
                          <th>Question</th>
                          <th>End date</th>
                          <th className="numeric-col">Market price</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.open_markets.map((market) => (
                          <tr key={`market-price-${market.id}`}>
                            <td>
                              <div className="probability-question">
                                <strong>{market.question}</strong>
                                {market.slug && (
                                  <a
                                    href={`https://polymarket.com/market/${market.slug}`}
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    {market.slug}
                                  </a>
                                )}
                              </div>
                            </td>
                            <td>{market.end_date || "—"}</td>
                            <td className="numeric-col">
                              <span className="probability-badge">
                                {formatProbability(getYesProbability(market))}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <div className="detail-section">
                <h3>Simulation probabilities</h3>
                {detail.market_probabilities.length === 0 ? (
                  <div className="state">
                    No simulation probability estimates yet.
                  </div>
                ) : (
                  <div className="table-wrap">
                    <table className="compact-table probability-table">
                      <thead>
                        <tr>
                          <th>Question</th>
                          <th>Implication date</th>
                          <th className="numeric-col">Simulation average</th>
                          <th className="numeric-col">Samples</th>
                          <th>Generated</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.market_probabilities.map((estimate) => (
                          <tr key={`${estimate.market_id}-${estimate.generated_at}`}>
                            <td>
                              <div className="probability-question">
                                <strong>{estimate.market_question}</strong>
                                {estimate.market_slug && (
                                  <a
                                    href={`https://polymarket.com/market/${estimate.market_slug}`}
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    {estimate.market_slug}
                                  </a>
                                )}
                              </div>
                            </td>
                            <td>{estimate.implication_date}</td>
                            <td className="numeric-col">
                              <span className="probability-badge">
                                {formatProbability(estimate.probability)}
                              </span>
                            </td>
                            <td className="numeric-col">{estimate.samples}</td>
                            <td>{estimate.generated_at}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <div className="detail-section">
                <h3>OpenForecaster estimates</h3>
                {detail.openforecaster_probabilities.length === 0 ? (
                  <div className="state">No OpenForecaster estimates yet.</div>
                ) : (
                  <div className="table-wrap">
                    <table className="compact-table probability-table">
                      <thead>
                        <tr>
                          <th>Question</th>
                          <th>End date</th>
                          <th className="numeric-col">Probability</th>
                          <th>Generated</th>
                          <th>Model</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.openforecaster_probabilities.map((estimate) => (
                          <tr
                            key={`${estimate.market_id}-${estimate.generated_at}`}
                          >
                            <td>
                              <div className="probability-question">
                                <strong>{estimate.market_question}</strong>
                                {estimate.market_slug && (
                                  <a
                                    href={`https://polymarket.com/market/${estimate.market_slug}`}
                                    target="_blank"
                                    rel="noreferrer"
                                  >
                                    {estimate.market_slug}
                                  </a>
                                )}
                              </div>
                            </td>
                            <td>{estimate.market_end_date || "—"}</td>
                            <td className="numeric-col">
                              <span className="probability-badge">
                                {formatProbability(
                                  estimate.estimated_probability
                                )}
                              </span>
                            </td>
                            <td>{estimate.generated_at}</td>
                            <td>{estimate.model}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <div className="detail-section">
                <h3>Relevant news</h3>
                {detail.relevant_news.length === 0 ? (
                  <div className="state">No relevant news captured yet.</div>
                ) : (
                  <div className="news-list">
                    {detail.relevant_news.map((article) => (
                      <article
                        key={`${article.article_id}-${article.run_id}`}
                        className="news-card"
                      >
                        <div>
                          <a
                            href={article.article_url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {article.article_title}
                          </a>
                          <p>{article.article_url}</p>
                        </div>
                        <div className="news-meta">
                          <span>{article.article_published_at}</span>
                          <span>{article.model}</span>
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </div>

              <div className="detail-section">
                <h3>Timelines</h3>
                {detail.present_timeline ? (
                  <div className="timeline-card">
                    <div className="timeline-header">
                      <strong>Present timeline</strong>
                      <span>{detail.present_timeline.generated_at}</span>
                    </div>
                    <pre className="timeline-text">
                      {detail.present_timeline.summary}
                    </pre>
                  </div>
                ) : (
                  <div className="state">No present timeline stored.</div>
                )}
                {latestFutureTimelines.length > 0 && (
                  <div className="timeline-stack">
                    {latestFutureTimelines.map((timeline, index) => (
                      <div
                        key={`${timeline.generated_at}-${index}`}
                        className="timeline-card"
                      >
                        <div className="timeline-header">
                          <strong>Future timeline</strong>
                          <span>{timeline.generated_at}</span>
                        </div>
                        <p className="timeline-scenario">
                          {timeline.scenario}
                        </p>
                        {timeline.results.map((result, resultIndex) => (
                          <div
                            key={`${timeline.generated_at}-${resultIndex}`}
                            className="timeline-rollout"
                          >
                            <pre className="timeline-text">
                              {result.simulated_timeline}
                            </pre>
                            {detail.open_markets.length > 0 && (
                              <div className="timeline-implications">
                                {detail.open_markets.map((market) => {
                                  const implication =
                                    result.market_implications.find(
                                      (item) => item.market_id === market.id
                                    );
                                  const outcome = implication
                                    ? implication.implied_answer
                                      ? "Yes"
                                      : "No"
                                    : "n/a";
                                  const outcomeClass = implication
                                    ? implication.implied_answer
                                      ? "is-yes"
                                      : "is-no"
                                    : "is-unknown";
                                  const label = market.slug || market.question;
                                  return (
                                    <div
                                      key={`${timeline.generated_at}-${resultIndex}-${market.id}`}
                                      className="timeline-implication-chip"
                                      title={market.question}
                                    >
                                      <span className="timeline-implication-label">
                                        {label}
                                      </span>
                                      <span
                                        className={`timeline-implication-outcome ${outcomeClass}`}
                                      >
                                        {outcome}
                                      </span>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
