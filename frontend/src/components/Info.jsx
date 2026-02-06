import {useInfo} from "../hooks/useInfo.js";

function Info({ tickers, info = "" }) {
    const {data, loading, error} = useInfo(tickers, info)

    if (loading) return <div className="text-secondary p-3">Loading details...</div>;
    if (error) return <div className="text-danger p-3">Error: {error}</div>;
    if (!data || Object.keys(data).length === 0) return <div className="p-3">No data available</div>;

    // Helper to render the Key-Value list (for "info" or "my_chart")
    const renderKeyValueTable = (obj) => (
        <table className="table table-striped table-sm">
            <tbody>
                {Object.entries(obj).map(([key, val]) => (
                    <tr key={key}>
                        <td className="fw-bold text-secondary" style={{width: '40%'}}>{key}</td>
                        <td>
                            {typeof val === 'object' && val !== null
                                ? JSON.stringify(val)
                                : val?.toString()}
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
    );

    // Helper to render the Data Grid (for "financials" or "balance_sheet")
    const renderDataTable = (arr) => {
        if (arr.length === 0) return <div>No records found.</div>;

        // Dynamic headers based on the keys of the first item
        const columns = Object.keys(arr[0]);

        return (
            <div className="table-responsive">
                <table className="table table-bordered table-sm table-hover">
                    <thead className="table-light">
                        <tr>
                            {columns.map(col => <th key={col}>{col}</th>)}
                        </tr>
                    </thead>
                    <tbody>
                        {arr.map((row, idx) => (
                            <tr key={idx}>
                                {columns.map(col => (
                                    <td key={col}>
                                        {/* Handle potential nested objects or nulls cleanly */}
                                        {typeof row[col] === 'object' && row[col] !== null
                                            ? JSON.stringify(row[col])
                                            : row[col]}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        );
    };

    return (
        <div className="card shadow-sm mt-3">
            <div className="card-header bg-white">
                 <h5 className="card-title text-capitalize mb-0 text-primary">
                    {info === "" ? "General Info" : info.replace("_", " ")}
                </h5>
            </div>
            <div className="card-body">

                {Object.entries(data).map(([ticker, content]) => (
                    <div key={ticker} className="mb-5">
                        <h4 className="border-bottom pb-2 mb-3">{ticker}</h4>

                        {Array.isArray(content)
                            ? renderDataTable(content)
                            : typeof content === 'object' && content !== null
                                ? renderKeyValueTable(content)
                                : <div>No valid data structure found</div>
                        }
                    </div>
                ))}
            </div>
        </div>
    );
}

export default Info;