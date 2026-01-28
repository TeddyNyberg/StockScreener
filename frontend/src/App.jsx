import TopBanner from "./components/TopBanner.jsx";
import "./App.css";
import StockChart from "./components/StockChart.jsx";
import { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom"; // Import everything here
import Watchlist from "./components/Watchlist.jsx";
import Portfolio from "./components/Portfolio.jsx";
import Details from "./components/Details.jsx";
import Home from "./components/Home.jsx";

function MainContent() {

    const navigate = useNavigate();

    async function navDetailsPage(tickerString) {
        navigate(`/details/${tickerString}`);
    }

    async function fetchTickerData(tickerString, time){
        try {
            const response = await fetch(`http://localhost:8000/chart?tickers=${tickerString}&time=${time}`);
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

    return (
        <>
            <TopBanner onSearch={navDetailsPage} />
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/watchlist" element={<Watchlist />} />
                <Route path="/portfolio" element={<Portfolio />} />
                <Route path="/details/:ticker" element={<Details />}/>
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