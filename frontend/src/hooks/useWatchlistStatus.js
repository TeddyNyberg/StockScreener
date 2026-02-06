import {useEffect, useState} from "react";
import {useAuth} from "../context/AuthContext.jsx";
import {apiRequest} from "../utils/api.js";

export function useWatchlistStatus({ticker}) {
    const {user, setShowLogin} = useAuth();
    const [inWatchlist, setInWatchlist] = useState(false);

    useEffect(() => {
        if (!user || !ticker) {
            setInWatchlist(false);
            return;
        }

        apiRequest(`/watchlist/check/${ticker}`)
            .then(data => {
                setInWatchlist(data.in_watchlist);
            })
            .catch(err => {
                console.error("Error checking watchlist:", err);
                setInWatchlist(false);
            });
    }, [user, ticker]);

    function toggleWatchlist() {
        if (!user) {
            setShowLogin(true);
            return;
        }

        const endpoint = inWatchlist ? "/watchlist/remove" : "/watchlist/add";
        setInWatchlist(!inWatchlist);

        apiRequest(`${endpoint}`, {
            method: "POST",
            body: JSON.stringify({ticker: ticker})
        })
            .catch(err => {
                setInWatchlist(!inWatchlist);
                console.error(err);
            });
    }
    return {inWatchlist, toggleWatchlist};
}