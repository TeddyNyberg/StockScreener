import {useEffect, useState} from "react";

export function useInfo(tickers, info = "" ){
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!tickers) return;

        setLoading(true);
        setError(null);
        setData(null);

        fetch(`http://localhost:8000/info?tickers=${tickers}&info=${info}`)
            .then(res => {
                if (!res.ok) throw new Error("Network response was not ok");
                return res.json();
            })
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