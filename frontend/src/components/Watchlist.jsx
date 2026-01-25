import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

function Watchlist() {
    const [stocks, setStocks] = useState([]);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const token = localStorage.getItem('token');

        // If no token even exists locally, don't even bother fetching
        if (!token) {
            navigate("/"); // or to login
            return;
        }

        fetch("http://localhost:8000/watchlist", {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            }
        })
        .then(res => {
            if (res.status === 401) {
                throw new Error("Unauthorized");
            }
            return res.json();
        })
        .then(data => {
            setStocks(data); // This is your ticker_list from Python
        })
        .catch(err => {
            console.error(err);
            if (err.message === "Unauthorized") {
                localStorage.removeItem('token'); // Clear bad token
                navigate("/"); // Send them home
            }
            setError("Failed to load watchlist.");
        });
    }, [navigate]);

    return (
        <div className="container mt-4">
            <h1>Your Watchlist</h1>
            <hr />
            {error && <div className="alert alert-danger">{error}</div>}

            {stocks.length === 0 ? (
                <p>Your watchlist is empty. Add some stocks from the home page!</p>
            ) : (
                <ul className="list-group">
                    {stocks.map(ticker => (
                        <li key={ticker} className="list-group-item d-flex justify-content-between">
                            <strong>{ticker}</strong>
                            <button className="btn btn-sm btn-outline-danger">Remove</button>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
export default Watchlist;