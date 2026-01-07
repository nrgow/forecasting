import { useEffect, useMemo, useState } from "react";
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable
} from "@tanstack/react-table";

const EMPTY_LIST = [];

export default function App() {
  const [rows, setRows] = useState(EMPTY_LIST);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");
  const [sorting, setSorting] = useState([]);

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

  const columns = useMemo(() => {
    if (rows.length === 0) {
      return [];
    }

    const keys = Object.keys(rows[0]);
    const focusKeys = ["event_names", "event_urls"];
    const baseKeys = keys.filter((key) => !focusKeys.includes(key));
    const orderedKeys = [...baseKeys];

    if (keys.includes("event_names")) {
      orderedKeys.splice(1, 0, "event_names");
    }

    if (keys.includes("event_urls")) {
      orderedKeys.splice(2, 0, "event_urls");
    }

    return orderedKeys.map((key) => ({
      accessorKey: key,
      header: key,
      cell: (info) => formatCellValue(info.getValue()),
      enableSorting: true
    }));
  }, [rows]);

  const table = useReactTable({
    data: rows,
    columns,
    state: {
      sorting
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel()
  });

  function formatCellValue(value) {
    if (value === null || value === undefined) {
      return "";
    }

    if (typeof value === "number") {
      if (Number.isInteger(value)) {
        return String(value);
      }
      return String(Number(value.toFixed(2)));
    }

    if (typeof value === "object") {
      return JSON.stringify(value);
    }

    return String(value);
  }

  return (
    <div className="page">
      <header className="terminal-bar">
        <div className="terminal-brand">
          <div className="brand-badge">NT</div>
          <div>
            <p className="brand-title">NEWSTALK TERMINAL</p>
            <p className="brand-subtitle">Event stats feed</p>
          </div>
        </div>
      </header>

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
          <div className="table-wrap">
            <table>
              <thead>
                {table.getHeaderGroups().map((headerGroup) => (
                  <tr key={headerGroup.id}>
                    {headerGroup.headers.map((header) => (
                      <th key={header.id} data-col={header.column.id}>
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
                      <td key={cell.id} data-col={cell.column.id}>
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
        )}
      </section>
    </div>
  );
}
