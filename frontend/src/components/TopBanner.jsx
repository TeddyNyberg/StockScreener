import SearchBar from "./SearchBar.jsx";
import "./TopBanner.css"
import { useState } from "react";
import LoginModal from "./LoginModal.jsx";
import {Link, useNavigate} from "react-router-dom";



function TopBanner({onSearch}) {
    const [showLogin, setShowLogin] = useState(false);
    const [user, setUser] = useState(() => sessionStorage.getItem('username'));

    const navigate = useNavigate();

    function handleWatchlistClick(){
        const token = sessionStorage.getItem('token');
        if(!token){
            setShowLogin(true);
            return;
        }
        navigate("/watchlist");
    }

    function handlePortfolioClick(){
        const token = sessionStorage.getItem('token');
        if(!token){
            setShowLogin(true);
            return;
        }
        navigate("/portfolio");
    }

    function handleModelClick(){
        const token = sessionStorage.getItem('token');
        if(!token){
            setShowLogin(true);
            return;
        }
        navigate("/model");
    }


    return (
        <>
        <nav className="navbar navbar-expand-lg navbar-dark bg-dark px-4 py-3">
            <Link className="col navbar-brand fw-bold fs-3" to="/">Nyberg.grq</Link>
            <div className="col collapse navbar-collapse justify-content-center">
                <div className="navbar-nav gap-3">
                    <button className="nav-link btn btn-link the-buttons" onClick={handleModelClick}>Model</button>
                    <button className="nav-link btn btn-link the-buttons" onClick={handleWatchlistClick}>Watchlist</button>
                    <button className="nav-link btn btn-link the-buttons" onClick={handlePortfolioClick}>Investments</button>
                </div>
            </div>

            <div className="col d-flex align-items-center gap-3">
                <div style={{ width: '250px' }}>
                    <SearchBar onSearch={onSearch}/>
                </div>
                {user ? (
                        <span className="text-white fw-bold">{user}</span>
                    ) : (
                        <button
                            className="nav-link btn btn-link the-buttons"
                            onClick={() => setShowLogin(true)}
                        >
                            Sign in
                        </button>
                    )}

            </div>
        </nav>
            <LoginModal
                show={showLogin}
                onClose={() => setShowLogin(false)}
                onLoginSuccess={(u) => setUser(u)}
            />

    </>
    );
}

export default TopBanner;