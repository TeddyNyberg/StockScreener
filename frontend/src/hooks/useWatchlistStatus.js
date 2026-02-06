import {useEffect, useState} from "react";
import {useAuth} from "../context/AuthContext.jsx";

export function useWatchlistStatus(ticker){
    const { user, setShowLogin } = useAuth();
    const [inWatchlist, setInWatchlist] = useState(false);

    useEffect(() => {
        const token = sessionStorage.getItem('token');

        if (user && token) {
            fetch(`http://localhost:8000/watchlist/check/${ticker}`, {
                headers: { "Authorization": `Bearer ${token}` }
            })
            .then(res => res.json())
            .then(data => {
                setInWatchlist(data.in_watchlist);
            })
            .catch(err => console.error("Error checking watchlist:", err));
        } else {
            setInWatchlist(false);
        }
    }, [user, ticker]);

    function toggleWatchlist() {
        const token = sessionStorage.getItem('token');
        if (!user || !token) {
            setShowLogin(true);
            return;
        }

        const endpoint = inWatchlist ? "/watchlist/remove" : "/watchlist/add";

        setInWatchlist(!inWatchlist);

        fetch(`http://localhost:8000${endpoint}`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ ticker: ticker })
        })
        .then(res => {
            if (!res.ok) {
                setInWatchlist(!inWatchlist);
                console.error("Failed to update watchlist");
            }
        })
        .catch(err => {
            setInWatchlist(!inWatchlist);
            console.error(err);
        });
    }

    return { inWatchlist, toggleWatchlist };
}