import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

function Watchlist() {
    const [data, setData] = useState([]);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    function getColor(val) {
        if (val > 0) return "text-success";
        if (val < 0) return "text-danger";
        return "text-muted";
    }

    useEffect(() => {
        const token = sessionStorage.getItem('token');

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
        .then(apiData => {
            setData(apiData); // This is your ticker_list from Python
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

            {data.length === 0 ? (
                <p>Your watchlist is empty. Add some stocks from the home page!</p>
            ) : (
                <ul className="list-group">
                    {data.map(item => (
                        <li key={item.ticker} className="list-group-item d-flex justify-content-between align-items-center">
                            <div>
                                <h5 className="mb-0">{item.ticker}</h5>
                            </div>

                            <div className="text-end">
                                <div className="fw-bold">
                                    ${item.price ? item.price.toFixed(2) : "---"}
                                </div>

                                <div className={`small ${getColor(item.change)}`}>
                                    {item.change > 0 ? "+" : ""}
                                    {item.change ? item.change.toFixed(2) : "-"}

                                    <span className="ms-1">
                                        ({item.change_percent ? item.change_percent.toFixed(2) : "-"}%)
                                    </span>
                                </div>
                            </div>

                            <button className="btn btn-sm btn-outline-danger ms-3">X</button>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
export default Watchlist;