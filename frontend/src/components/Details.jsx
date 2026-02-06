import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import StockChart from "./StockChart";
import Info from "./Info.jsx";
import { useAuth } from "../context/AuthContext.jsx";
import "./Details.css"
import TradeModal from "./TradeModal.jsx";

function Details() {
    const { tickers } = useParams();
    const [chartData, setChartData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [timeRange, setTimeRange] = useState("1Y");
    const [searchParams, setSearchParams] = useSearchParams();

    const { user, setShowLogin } = useAuth();
    const [inWatchlist, setInWatchlist] = useState(false);

    const currentView = searchParams.get("info") || "";
    const timeButtons = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"];
    const currentTicker = tickers.split(",")[0];

    const [showTradeModal, setShowTradeModal] = useState(false);

    const VIEWS = [
        {label: "General Info", value: ""},
        {label: "Financials", value: "financials"},
        {label: "Balance Sheet", value: "balance_sheet"},
        {label: "Focused View", value: "my_chart"},
    ];

    const handleViewChange = (newView) => {
        setSearchParams({ info: newView });
    };

    function handleTradeClick(){
        if(!user){
            setShowLogin(true);
            return;
        }
        setShowTradeModal(true);
    }


    function toggleWatchlist() {
        const token = sessionStorage.getItem('token');

        if (!user || !token) {
            setShowLogin(true);
            return;
        }

        const endpoint = inWatchlist ? "/watchlist/remove" : "/watchlist/add";

        setInWatchlist(!inWatchlist);

        fetch(`http://localhost:8000${endpoint}`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ ticker: currentTicker })
        })
        .then(res => {
            if (!res.ok) {
                setInWatchlist(!inWatchlist);
                console.error("Failed to update watchlist");
            }
        })
        .catch(err => {
            setInWatchlist(!inWatchlist);
            console.error(err);
        });
    }

    useEffect(() => {
        const token = sessionStorage.getItem('token');

        if (user && token) {
            fetch(`http://localhost:8000/watchlist/check/${currentTicker}`, {
                headers: { "Authorization": `Bearer ${token}` }
            })
            .then(res => res.json())
            .then(data => {
                setInWatchlist(data.in_watchlist);
            })
            .catch(err => console.error("Error checking watchlist:", err));
        } else {
            setInWatchlist(false);
        }
    }, [user, currentTicker]);

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
            <TradeModal
                show={showTradeModal}
                onHide={() => setShowTradeModal(false)}
                ticker={currentTicker}
                currentPrice={chartData?.[currentTicker]?.at(-1)?.Close || 0}
                user={user}
            />
            <div className="row align-items-center">
                <div className="col">
                    <h1 className="mb-0">{tickers.toUpperCase()}</h1>
                </div>

                <div className="col-auto d-flex gap-2">
                    <button
                        className={`btn rect-btn ${inWatchlist ? "btn-success" : ""}`}
                        onClick={toggleWatchlist}
                        style={inWatchlist ? { color: "#4ade80", borderColor: "#4ade80" } : {}}
                    >
                        {inWatchlist ? "Watching ✓" : "Watchlist +"}
                    </button>

                    <button className="btn rect-btn" onClick={handleTradeClick}>
                        Trade
                    </button>
                </div>
            </div>



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