import {usePortfolio} from "../hooks/usePortfolio.js";
import PortfolioItem from "./PortfolioItem.jsx";

function Portfolio() {
    const { portfolio, error } = usePortfolio();

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
                                <PortfolioItem key={index} item={item}></PortfolioItem>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}

export default Portfolio;