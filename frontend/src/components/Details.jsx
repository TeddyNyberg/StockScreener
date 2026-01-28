import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import StockChart from "./StockChart";

function Details() {
    const { ticker } = useParams();
    const [chartData, setChartData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);


    useEffect(() => {
        setLoading(true);
        fetch(`http://localhost:8000/chart?tickers=${ticker}&time=1Y`)
            .then(res => {
                if (!res.ok) throw new Error("Could not find stock");
                return res.json();
            })
            .then(data => {
                setChartData(data);
                setLoading(false);
            })
            .catch(err => {
                setError(err.message);
                setLoading(false);
            });
    }, [ticker]);

    return (
        <div className="container mt-4">
            <h1>Analysis: {ticker.toUpperCase()}</h1>

            {loading && <div className="spinner-border text-primary"></div>}

            {error && <div className="alert alert-danger">{error}</div>}

            {!loading && !error && chartData && (
                <StockChart
                    apiData={chartData}
                    tickers={ticker.split(",")}
                />
            )}
        </div>
    );
}

export default Details;