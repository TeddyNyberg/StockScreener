import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { useSearchParams } from "react-router-dom";
import StockChart from "./StockChart";
import Info from "./Info.jsx";

function Details() {
    const { tickers } = useParams();
    const [chartData, setChartData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [timeRange, setTimeRange] = useState("1Y");
    const [searchParams, setSearchParams] = useSearchParams();

    const currentView = searchParams.get("info") || "";

    const timeButtons = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"];

    const VIEWS = [
        {label: "General Info", value: ""},
        {label: "Financials", value: "financials"},
        {label: "Balance Sheet", value: "balance_sheet"},
        {label: "Focused View", value: "my_chart"},
    ];

    const handleViewChange = (newView) => {
        setSearchParams({ info: newView });
    };

    useEffect(() => {
        setLoading(true);
        fetch(`http://localhost:8000/chart?tickers=${tickers}&time=${timeRange}`)
            .then(res => {
                if (!res.ok) throw new Error("Could not find stock");
                return res.json();
            })
            .then(data => {
                setChartData(data);
                setLoading(false);
            })
            .catch(err => {
                setError(err.message);
                setLoading(false);
            });
    }, [tickers, timeRange]);


    return (
        <div className="container mt-4">
            <h1>Analysis: {tickers.toUpperCase()}</h1>

            {error && <div className="alert alert-danger">{error}</div>}

            {loading && !chartData && <div className="spinner-border text-primary"></div>}

            {chartData && (
                <>
                    <div style={{ opacity: loading ? 0.5 : 1, transition: 'opacity 0.2s' }}>
                        <StockChart apiData={chartData} tickers={tickers.split(",")} />
                    </div>

                    <div className="my-3">
                        {timeButtons.map(time => (
                            <button
                                key={time}
                                disabled={loading}
                                style={{ margin: "2px" }}
                                className={`btn btn-sm ${timeRange === time ? 'btn-primary' : 'btn-outline-secondary'}`}
                                onClick={() => setTimeRange(time)}>
                                {time}
                            </button>
                        ))}
                        {loading && <span className="spinner-border spinner-border-sm ms-2 text-secondary"></span>}
                    </div>

                    <div className="btn-group mb-4 shadow-sm" role="group">
                        {VIEWS.map((view) => (
                            <button
                                key={view.label}
                                type="button"
                                className={`btn ${currentView === view.value ? "btn-primary" : "btn-outline-primary"}`}
                                onClick={() => handleViewChange(view.value)}
                            >
                                {view.label}
                            </button>
                        ))}
                    </div>
                    <Info tickers={tickers} info={currentView} />
                </>
            )}

        </div>
    );
}

export default Details;