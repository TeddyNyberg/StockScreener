import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

function Portfolio() {
    const [portfolio, setPortfolio] = useState([]);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    // color positive/negative numbers
    function getColor(val) {
        if (val > 0) return "text-success";
        if (val < 0) return "text-danger";
        return "text-muted";
    }

    useEffect(() => {
        const token = sessionStorage.getItem('token');

        if (!token) {
            navigate("/"); // Redirect to login/home if no token
            return;
        }

        fetch("http://localhost:8000/portfolio", {
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
            if (!res.ok) {
                throw new Error("Failed to fetch data");
            }
            return res.json();
        })
        .then(apiData => {
            setPortfolio(apiData);
        })
        .catch(err => {
            console.error(err);
            if (err.message === "Unauthorized") {
                sessionStorage.removeItem('token');
                navigate("/");
            } else {
                setError("Unable to load portfolio data.");
            }
        });
    }, [navigate]);

    const totalAccountValue = portfolio.reduce((acc, item) => acc + (item.market_value || 0), 0);

    return (
        <div className="container mt-4">
            <div className="d-flex justify-content-between align-items-center mb-3">
                <h1>My Portfolio</h1>
                <h3>Total Value: <span className="text-primary">${totalAccountValue.toFixed(2)}</span></h3>
            </div>

            <hr />

            {error && <div className="alert alert-danger">{error}</div>}

            {portfolio.length === 0 && !error ? (
                <div className="text-center mt-5">
                    <p className="lead">You don't have any positions yet.</p>
                </div>
            ) : (
                <div className="table-responsive">
                    <table className="table table-hover table-striped">
                        <thead className="table-dark">
                            <tr>
                                <th>Ticker</th>
                                <th className="text-end">Shares</th>
                                <th className="text-end">Avg Cost</th>
                                <th className="text-end">Current Price</th>
                                <th className="text-end">Market Value</th>
                                <th className="text-end">Total Return ($)</th>
                                <th className="text-end">Total Return (%)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {portfolio.map((item, index) => (
                                <tr key={index}>
                                    <td className="fw-bold">{item.ticker}</td>

                                    <td className="text-end">{item.shares}</td>

                                    <td className="text-end">
                                        ${item.avg ? item.avg.toFixed(2) : "0.00"}
                                    </td>

                                    <td className="text-end">
                                        ${item.price ? item.price.toFixed(2) : "0.00"}
                                    </td>

                                    <td className="text-end fw-bold">
                                        ${item.market_value ? item.market_value.toFixed(2) : "0.00"}
                                    </td>

                                    <td className={`text-end ${getColor(item.pl)}`}>
                                        {item.pl > 0 ? "+" : ""}
                                        {item.pl ? item.pl.toFixed(2) : "0.00"}
                                    </td>

                                    <td className={`text-end ${getColor(item.pct)}`}>
                                        {item.pct > 0 ? "+" : ""}
                                        {item.pct ? item.pct.toFixed(2) : "0.00"}%
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}

export default Portfolio;