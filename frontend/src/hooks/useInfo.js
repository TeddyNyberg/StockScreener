import {useEffect, useState} from "react";
import {apiRequest} from "../utils/api.js";

export function useInfo({tickers, info = "" }){
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!tickers) return;

        setLoading(true);
        setError(null);
        setData(null);

        apiRequest(`/info?tickers=${tickers}&info=${info}`)
            .then(fetchedData => {
                setData(fetchedData);
                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setError(err.message);
                setLoading(false);
            });

    }, [tickers, info]);

    return {data, loading, error};
}