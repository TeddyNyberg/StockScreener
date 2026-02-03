import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import StockChart from "./StockChart";
import Info from "./Info.jsx";

function Details() {
    const { tickers } = useParams();
    const [chartData, setChartData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [timeRange, setTimeRange] = useState("1Y");

    const timeButtons = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"];

    useEffect(() => {
        setLoading(true);
        fetch(`http://localhost:8000/chart?tickers=${tickers}&time=${timeRange}`)
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
    }, [tickers, timeRange]);


    return (
        <div className="container mt-4">
            <h1>Analysis: {tickers.toUpperCase()}</h1>

            {loading && <div className="spinner-border text-primary"></div>}

            {error && <div className="alert alert-danger">{error}</div>}

            {!loading && !error && chartData && (
                <>
                    <StockChart apiData={chartData} tickers={tickers.split(",")}/>
                    <div>
                        {timeButtons.map(time => (
                                <button
                                    key={time}
                                    style={{margin:"2px"}}
                                    className={`btn btn-sm ${timeRange === time ? 'btn-primary' : 'btn-outline-secondary'}`}
                                    onClick={() => setTimeRange(time)}>
                                    {time}
                                </button>
                            )
                        )}
                    </div>
                    <Info tickers={tickers} info={""}/>
                </>
            )}

        </div>
    );
}

export default Details;