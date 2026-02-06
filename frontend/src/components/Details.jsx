import { useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import StockChart from "./StockChart";
import Info from "./Info.jsx";
import "./Details.css"
import TradeModal from "./TradeModal.jsx";
import TimeFrameSelector from "./TimeFrameSelector.jsx";
import ViewSelector from "./ViewSelector.jsx";
import {useWatchlistStatus} from "../hooks/useWatchlistStatus.js";
import {useAuth} from "../context/AuthContext.jsx";
import {useStockChart} from "../hooks/useStockChart.js";

function Details() {
    const { tickers } = useParams();
    const [timeRange, setTimeRange] = useState("1Y");
    const [searchParams, setSearchParams] = useSearchParams();
    const [showTradeModal, setShowTradeModal] = useState(false);
    const currentView = searchParams.get("info") || "";
    const currentTicker = tickers.split(",")[0];
    const { inWatchlist, toggleWatchlist } = useWatchlistStatus({ticker:currentTicker});
    const { chartData, loading, error } = useStockChart({tickers: tickers, timeRange:timeRange});
    const { user, setShowLogin } = useAuth();

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

    return (
        <div className="container mt-4">
            <TradeModal
                show={showTradeModal}
                onHide={() => setShowTradeModal(false)}
                ticker={currentTicker}
                currentPrice={Number(chartData?.[currentTicker]?.at(-1)?.Close || 0).toFixed(2)}
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
                    <TimeFrameSelector timeRange={timeRange} setTimeRange={setTimeRange} loading={loading}/>
                    <ViewSelector currentView={currentView} handleViewChange={handleViewChange} />
                    <Info tickers={tickers} info={currentView} />
                </>
            )}

        </div>
    );
}

export default Details;