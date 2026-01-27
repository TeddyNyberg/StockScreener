import TopBanner from "./components/TopBanner.jsx";
import "./App.css";
import StockChart from "./components/StockChart.jsx";
import { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom"; // Import everything here
import Watchlist from "./components/Watchlist.jsx";
import Portfolio from "./components/Portfolio.jsx";

function MainContent() {
    const [chartData, setChartData] = useState(null);
    const [tickers, setTickers] = useState(["SPY"]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const navigate = useNavigate();

    async function fetchStockData(tickerString) {
        setError(null);
        setLoading(true);
        navigate("/");

        try {
            const response = await fetch(`http://localhost:8000/chart?tickers=${tickerString}&time=1Y`);

            if (!response.ok) {
                throw new Error("no bueno");
            }
            const data = await response.json();
            setChartData(data);
            setTickers(tickerString.split(",").map(t => t.trim().toUpperCase()));
        } catch (err) {
            setError("Failed to fetch data: " + err.message);
        } finally {
            setLoading(false);
        }
    }

    // Initial load
    useEffect(() => {
        fetchStockData("SPY").catch(console.error);
    }, []);

    return (
        <>
            <TopBanner onSearch={fetchStockData} />
            <div className="container mt-3">
                {error && <div className="alert alert-danger">{error}</div>}
                {loading && <div className="spinner-border text-primary" role="status"></div>}
            </div>

            <Routes>
                <Route path="/" element={<StockChart apiData={chartData} tickers={tickers} />} />
                <Route path="/watchlist" element={<Watchlist />} />
                <Route path="/portfolio" element={<Portfolio />} />
            </Routes>
        </>
    );
}

function App() {
    return (
        <Router>
            <MainContent />
        </Router>
    );
}

export default App;