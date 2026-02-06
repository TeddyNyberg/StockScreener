import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {useAuth} from "../context/AuthContext.jsx";
import {apiRequest} from "../utils/api.js";

export function useWatchlist() {
    const [data, setData] = useState([]);
    const [error, setError] = useState(null);
    const { user } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
        if (!user ) {
            navigate("/");
            return;
        }

        apiRequest("/watchlist")
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