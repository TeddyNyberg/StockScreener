import StockChart from "./StockChart.jsx";
import {useEffect, useState} from "react";

function Home(){
    const [chartData, setChartData] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {

        fetch("http://localhost:8000/chart?tickers=SPY&time=1Y")
            .then(res => res.json())
            .then(data => setChartData(data))
            .catch(err => setError("Failed to load SPY"));
    }, []);

    if (error) return <div className="alert alert-danger">{error}</div>;
    if (!chartData) return <div>Loading Market Data...</div>

    return (<div>
        <h2 className="text-center mt-3">Market Overview</h2>
        <StockChart apiData={chartData} tickers={["SPY"]} />
    </div>);
}
export default Home;