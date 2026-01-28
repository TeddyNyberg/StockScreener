import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export function useWatchlist() {
    const [data, setData] = useState([]);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const token = sessionStorage.getItem('token');
        if (!token) {
            navigate("/");
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
            if (res.status === 401) throw new Error("Unauthorized");
            return res.json();
        })
        .then(apiData => setData(apiData))
        .catch(err => {
            if (err.message === "Unauthorized") {
                sessionStorage.removeItem('token');
                navigate("/");
            }
            setError("Failed to load watchlist.");
        });
    }, [navigate]);

    return { data, error };
}