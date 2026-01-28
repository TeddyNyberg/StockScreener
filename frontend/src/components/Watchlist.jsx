
import WatchlistItem from "./WatchlistItem.jsx";
import {fetchWatchlist} from "../hooks/fetchWatchlist.js"

function Watchlist() {
    const { data, error } = fetchWatchlist();
    return (
        <div className="container mt-4">
            <h1>Your Watchlist</h1>
            <hr />
            {error && <div className="alert alert-danger">{error}</div>}

            {data.length === 0 ? (
                <p>Your watchlist is empty. Add some stocks from the home page!</p>
            ) : (
                <ul className="list-group">
                    {data.map(item => (
                        <WatchlistItem key={item.ticker} item={item}/>
                    ))}
                </ul>
            )}
        </div>
    );
}
export default Watchlist;