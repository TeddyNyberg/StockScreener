import {useEffect, useState} from "react";
import {apiRequest} from "../utils/api.js";

export function useTrading({show, ticker, currentPrice}){

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
            apiRequest(`/trade/info?ticker=${ticker}`)
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

        try {
            await apiRequest("/stock-transaction", {
                method: "POST",
                body: JSON.stringify({
                    ticker: ticker,
                    quantity: orderSize,
                    price: currentPrice,
                    type: orderType
                })
            });
            console.log(`${orderType} successful!`);
            onHide();
        } catch (err) {
            console.error("Transaction failed:", err);
            alert(`Trade failed: ${err.message}`);
        }
    }

    return {tradeData, orderSize, setOrderSize, orderType, setOrderType, handleTrade}
}