import {useEffect, useState} from "react";

export function useStockChart(tickers, timeRange){
    const [chartData, setChartData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        setLoading(true);
        setError(null)
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
    return {chartData, loading, error};
}