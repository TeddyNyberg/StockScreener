import SearchBar from "./SearchBar.jsx";
import "./TopBanner.css"
import { useAuth } from "../context/AuthContext.jsx";
import {Link, useNavigate} from "react-router-dom";


function TopBanner({onSearch}) {
    const { user, setShowLogin } = useAuth();

    const navigate = useNavigate();

    function handleProtectedClick(path){
        if(!user){
            setShowLogin(true);
            return;
        }
        navigate(path);
    }

    return (
        <>
        <nav className="navbar navbar-expand-lg navbar-dark bg-dark px-4 py-3">
            <Link className="col navbar-brand fw-bold fs-3" to="/">Nyberg.grq</Link>
            <div className="col collapse navbar-collapse justify-content-center">
                <div className="navbar-nav gap-3">
                    <button className="nav-link btn btn-link the-buttons" onClick={() =>handleProtectedClick("/model")}>Model</button>
                    <button className="nav-link btn btn-link the-buttons" onClick={() => handleProtectedClick("/watchlist")}>Watchlist</button>
                    <button className="nav-link btn btn-link the-buttons" onClick={() => handleProtectedClick("/portfolio")}>Investments</button>
                </div>
            </div>

            <div className="col d-flex align-items-center gap-3">
                <div style={{ width: '250px' }}>
                    <SearchBar onSearch={onSearch}/>
                </div>
                {user ? (
                        <button
                            className="nav-link btn btn-link the-buttons"
                            onClick={() => setShowLogin(true)} // onclick popup logout
                        >
                            {user}
                        </button>
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


    </>
    );
}
export default TopBanner;