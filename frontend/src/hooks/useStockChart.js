import {useEffect, useState} from "react";
import {apiRequest} from "../utils/api.js";

export function useStockChart({tickers, timeRange}){
    const [chartData, setChartData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        setLoading(true);
        setError(null)
        apiRequest(`/chart?tickers=${tickers}&time=${timeRange}`)
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