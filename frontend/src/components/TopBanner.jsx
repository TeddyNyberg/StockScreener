import SearchBar from "./SearchBar.jsx";
import "./TopBanner.css"

function TopBanner() {
    return (
        <nav className="navbar navbar-expand-lg navbar-dark bg-dark px-4 py-3">

            <a className="navbar-brand fw-bold fs-3" href="#">Nyberg.grq</a>

            <div className="collapse navbar-collapse justify-content-center">
                <div className="navbar-nav gap-3">
                    <button className="nav-link btn btn-link the-buttons">Model</button>
                    <button className="nav-link btn btn-link the-buttons">Watchlist</button>
                    <button className="nav-link btn btn-link the-buttons">Investments</button>
                </div>
            </div>

            <div className="d-flex align-items-center gap-3">
                <div style={{ width: '250px' }}>
                    <SearchBar />
                </div>
                <button type="button" className="nav-link btn btn-link the-buttons">Sign in</button>
            </div>
        </nav>
    );
}

export default TopBanner;