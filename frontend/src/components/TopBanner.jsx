import SearchBar from "./SearchBar.jsx";
import "./TopBanner.css"

function TopBanner() {
    return(<>
        <div className="top-banner">
            <div className="row align-items-start">
                <div className="col">
                    Nyberg.grq
                </div>
                <div className="col" />
                <div className="col-auto ms-auto search-bar-container">
                    <SearchBar />
                </div>
                <div className="col">
                    <button type="button" className="btn btn-outline-light sign-in-btn">Sign in</button>
                </div>
            </div>
            <div className="row align-items-start">
                <div className="col" />
                <div className="col">
                    <button type="button" className="btn btn-outline-secondary">Model</button>
                </div>
                <div className="col">
                    <button type="button" className="btn btn-outline-secondary">Watchlist</button>
                </div>
                <div className="col">
                    <button type="button" className="btn btn-outline-secondary">Investments</button>
                </div>
            </div>
        </div>
    </>);
}

export default TopBanner;