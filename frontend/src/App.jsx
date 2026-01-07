import { useEffect, useMemo, useState } from "react";
import {
  flexRender,
  getCoreRowModel,
  useReactTable
} from "@tanstack/react-table";

const EMPTY_LIST = [];

function formatCellValue(value) {
  if (value === null || value === undefined) {
    return "";
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}

export default function App() {
  const [rows, setRows] = useState(EMPTY_LIST);
  const [status, setStatus] = useState("loading");
  const [error, setError] = useState("");

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

    return keys.map((key) => ({
      accessorKey: key,
      header: key,
      cell: (info) => formatCellValue(info.getValue())
    }));
  }, [rows]);

  const table = useReactTable({
    data: rows,
    columns,
    getCoreRowModel: getCoreRowModel()
  });

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Live ingest</p>
          <h1>Event Stats Table</h1>
          <p className="subtitle">
            FastAPI → TanStack Table. A real-time snapshot of your event
            telemetry.
          </p>
        </div>
        <div className="stats">
          <div>
            <span className="stat-label">Rows</span>
            <span className="stat-value">{rows.length}</span>
          </div>
          <div>
            <span className="stat-label">Columns</span>
            <span className="stat-value">{columns.length}</span>
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
                      <th key={header.id}>
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext()
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
                      <td key={cell.id}>
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
