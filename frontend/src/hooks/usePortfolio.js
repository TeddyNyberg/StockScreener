import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {apiRequest} from "../utils/api.js";
import {useAuth} from "../context/AuthContext.jsx";

export function usePortfolio() {
    const [portfolio, setPortfolio] = useState([]);
    const [error, setError] = useState(null);
    const navigate = useNavigate();
    const { user } = useAuth();

    useEffect(() => {

        if (!user) {
            navigate("/"); // Redirect to login/home if no token
            return;
        }

        apiRequest("/portfolio")
        .then(apiData => {
            setPortfolio(apiData);
        })
        .catch(err => {
            console.error(err);
            if (err.message === "Unauthorized") {
                sessionStorage.removeItem('token');
                navigate("/");
            } else {
                setError("Unable to load portfolio data.");
            }
        });
    }, [navigate]);

    return {portfolio, error};
}