import TopBanner from "./components/TopBanner.jsx";
import "./App.css";
import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom";
import Watchlist from "./components/Watchlist.jsx";
import Portfolio from "./components/Portfolio.jsx";
import Details from "./components/Details.jsx";
import Home from "./components/Home.jsx";
import Model from "./components/Model.jsx";
import LoginModal from "./components/LoginModal.jsx";
import {AuthProvider} from "./context/AuthContext.jsx";
import TestLogin from "./components/TestLogin.jsx";

function MainContent() {

    const navigate = useNavigate();

    async function searchTicker(tickerString) {
        navigate(`/details/${tickerString}`);
    }

    return (
        <>
            <TopBanner onSearch={searchTicker} />
            <LoginModal />
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/watchlist" element={<Watchlist />} />
                <Route path="/portfolio" element={<Portfolio />} />
                <Route path="/details/:tickers" element={<Details />}/>
                <Route path="/model" element={<Model />} />
            </Routes>

        </>
    );
}

function App() {
    return (
        <AuthProvider>
            <Router>
                <MainContent />
            </Router>
        </AuthProvider>
    );
}

export default App;