import {useEffect, useState} from "react";

export function useTrading(show, ticker, currentPrice){

    const [tradeData, setTradeData] = useState({
        bid: 0,
        ask: 0,
        sharesOwned: 0,
        avgCost: 0
    });
    const [orderSize, setOrderSize] = useState("");
    const [orderType, setOrderType] = useState("BUY");

    useEffect(() => {
        if (show && ticker) {
            const token = sessionStorage.getItem("token");
            fetch(`http://localhost:8000/trade/info?ticker=${ticker}`, {
                headers: { "Authorization": `Bearer ${token}` }
            })
                .then(res => res.json())
                .then(data => {
                    const shares = data.shares_owned || 0;
                    const totalCost = data.cost_basis || 0;

                    setTradeData({
                        bid: data.bid || currentPrice,
                        ask: data.ask || currentPrice,
                        sharesOwned: shares,
                        avgCost: shares > 0 ? (totalCost / shares) : 0
                    });
                })
                .catch(err => console.error("Error fetching trade info:", err));
        }
    }, [show, ticker, currentPrice]);

    async function handleTrade(onHide) {
        console.log(`Executing ${orderType} for ${orderSize} shares of ${ticker}`);
        const token = sessionStorage.getItem('token');

        fetch(`http://localhost:8000/stock-transaction`, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ ticker: ticker, quantity: orderSize, price: currentPrice, type: orderType })
        })
        .then(async res => {
            if (res.ok) {
                console.log(`${orderType} successful!`);
                onHide();
            } else {
                const errorData = await res.json();
                console.error("Transaction failed:", errorData.detail || "Unknown error");
                alert(`Trade failed: ${errorData.detail || "Server error"}`);
            }
        })
        .catch(err => {

            console.error(err);
        });
    }

    return {tradeData, orderSize, setOrderSize, orderType, setOrderType, handleTrade}
}