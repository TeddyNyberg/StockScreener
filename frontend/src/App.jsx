import TopBanner from "./components/TopBanner.jsx";
import "./App.css";
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";
import Watchlist from "./components/Watchlist.jsx";
import Portfolio from "./components/Portfolio.jsx";
import Details from "./components/Details.jsx";
import Home from "./components/Home.jsx";

function MainContent() {

    const navigate = useNavigate();

    async function searchTicker(tickerString) {
        navigate(`/details/${tickerString}`);
    }


    return (
        <>
            <TopBanner onSearch={searchTicker} />
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/watchlist" element={<Watchlist />} />
                <Route path="/portfolio" element={<Portfolio />} />
                <Route path="/details/:tickers" element={<Details />}/>
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