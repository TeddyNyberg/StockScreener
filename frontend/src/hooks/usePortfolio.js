import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export function usePortfolio() {
    const [portfolio, setPortfolio] = useState([]);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const token = sessionStorage.getItem('token');

        if (!token) {
            navigate("/"); // Redirect to login/home if no token
            return;
        }

        fetch("http://localhost:8000/portfolio", {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            }
        })
        .then(res => {
            if (res.status === 401) {
                throw new Error("Unauthorized");
            }
            if (!res.ok) {
                throw new Error("Failed to fetch data");
            }
            return res.json();
        })
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