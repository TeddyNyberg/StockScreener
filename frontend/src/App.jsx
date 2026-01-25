import TopBanner from "./components/TopBanner.jsx";
import "./App.css"
import StockChart from "./components/StockChart.jsx";
import {useEffect, useState} from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Watchlist from "./components/Watchlist.jsx";

function App(){

    const [chartData, setChartData] = useState(null);
    const [tickers, setTickers] = useState(["SPY"]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    async function fetchStockData(tickerString){
        setError(null);
        setLoading(true);
        try {
            const response = await fetch(`http://localhost:8000/chart?tickers=${tickerString}&time=1Y`);

            if (!response.ok) {
                throw new Error("no bueno");
            }
            const data = await response.json();
            setChartData(data);
            setTickers(tickerString.split(",").map(t => t.trim().toUpperCase()))
        } catch (err){
            setError("Failed to fetch data: " + err.message);
        } finally {
            setLoading(false)
        }

    }

    useEffect(() => {
        fetchStockData("SPY").catch(console.error);
    }, []);

    return <Router>
        <TopBanner onSearch={fetchStockData}/>
        <Routes>
            <Route path="/" element={<StockChart apiData={chartData} tickers={tickers}/>}/>
            <Route path="/watchlist" element={<Watchlist />}/>
        </Routes>
    </Router>
}

export default App;