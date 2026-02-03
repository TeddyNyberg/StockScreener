import { useEffect, useState } from "react";

function Info({ tickers, info = "" }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);


    useEffect(() => {
        setLoading(true);

        fetch(`http://localhost:8000/info?tickers=${tickers}&info=${info}`)
            .then(res => res.json())
            .then(fetchedData => {
                setData(fetchedData);
                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setLoading(false);
            });

    }, [tickers, info]);

    if (loading) return <div className="text-secondary">Loading details...</div>;
    if (!data || Object.keys(data).length === 0) return <div>No data available</div>;

    let content = null;

    if (info === "info" || info === "my_chart" || info === "") {

        content = (
            <div>
                {Object.entries(data).map(([ticker, values]) => (
                    <div key={ticker} className="mb-4">
                        <h6 className="text-primary fw-bold">{ticker}</h6>
                        <table className="table table-striped table-sm">
                            <tbody>
                                {values && typeof values === 'object' ? (
                                    Object.entries(values).map(([key, val]) => (
                                        <tr key={key}>
                                            <td className="fw-bold">{key}</td>
                                            <td>
                                                {typeof val === 'object' && val !== null
                                                    ? JSON.stringify(val)
                                                    : val?.toString()}
                                            </td>
                                        </tr>
                                    ))
                                ) : (
                                    <tr><td colSpan="2">No data for this ticker</td></tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                ))}
            </div>
        );
    } else {

        if (!Array.isArray(data)) return <div>Invalid Data Format</div>;

        const columns = Object.keys(data[0]);

        content = (
            <div className="table-responsive">
                <table className="table table-bordered table-sm">
                    <thead>
                        <tr>
                            {columns.map(col => <th key={col}>{col}</th>)}
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((row, idx) => (
                            <tr key={idx}>
                                {columns.map(col => (
                                    <td key={col}>{row[col]}</td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        );
    }

    return (
        <div className="card p-3 shadow-sm mt-3">
            <h5 className="card-title text-capitalize">{info.replace("_", " ")}</h5>
            {content}
        </div>
    );
}

export default Info;