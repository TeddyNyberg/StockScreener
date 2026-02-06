import {useTrading} from "../hooks/useTrading.js";

function TradeModal({ show, onHide, ticker, currentPrice }) {

    const {
        tradeData, orderSize, setOrderSize,
        orderType, setOrderType, handleTrade
    } = useTrading(show, ticker, currentPrice);

    if (!show) return null;

    const estimatedTotal = (Number(orderSize) * (orderType === "BUY" ? tradeData.ask : tradeData.bid)).toFixed(2);

    return (
        <div className="modal show d-block" tabIndex="-1" style={{ backgroundColor: 'rgba(0,0,0,0.8)' }}>
            <div className="modal-dialog modal-dialog-centered">
                <div className="modal-content bg-white text-dark shadow-lg">

                    {/* Header */}
                    <div className="modal-header border-secondary">
                        <h5 className="modal-title">Trade {ticker}</h5>
                        <button type="button" className="btn-close" onClick={onHide}></button>
                    </div>
                    {/* HEREEEEE */}
                        <div className="modal-body">
                        <div className="row mb-3 text-center">
                            <div className="col">
                                <small className="text-muted d-block">Bid</small>
                                <strong>${tradeData.bid}</strong>
                            </div>
                            <div className="col">
                                <small className="text-muted d-block">Ask</small>
                                <strong>${tradeData.ask}</strong>
                            </div>
                            <div className="col">
                                <small className="text-muted d-block">Last</small>
                                <strong>${currentPrice}</strong>
                            </div>
                        </div>

                        {/* Position Info */}
                        <div className="p-3 mb-3 rounded" >
                            <div className="d-flex justify-content-between">
                                <span>Shares Owned:</span>
                                <strong>{tradeData.sharesOwned}</strong>
                            </div>
                            <div className="d-flex justify-content-between">
                                <span>Avg Cost:</span>
                                <strong>${tradeData.avgCost.toFixed(2)}</strong>
                            </div>
                        </div>

                        {/* Order Type Toggle */}
                        <div className="mb-3">
                            <label className="form-label small text-muted">Order Type</label>
                            <div className="d-flex gap-2">
                                <button
                                    className={`btn flex-grow-1 ${orderType === 'BUY' ? 'btn-primary' : 'btn-outline-primary'}`}
                                    onClick={() => setOrderType("BUY")}
                                >Buy</button>
                                <button
                                    className={`btn flex-grow-1 ${orderType === 'SELL' ? 'btn-danger' : 'btn-outline-danger'}`}
                                    onClick={() => setOrderType("SELL")}
                                >Sell</button>
                            </div>
                        </div>

                        {/* Quantity Input */}
                            <div className="mb-3">
                                <label className="form-label small fw-bold text-muted text-uppercase"
                                       style={{fontSize: '0.7rem'}}>
                                    Number of Shares
                                </label>
                                <input
                                    type="number"
                                    className="form-control border text-dark bg-white"
                                    placeholder="0"
                                    value={orderSize}
                                    style={{height: '45px', borderRadius: '8px'}}
                                    onChange={(e) => setOrderSize(e.target.value)}
                                />
                            </div>

                        <div className="text-end">
                            <span className="text-muted small">Estimated Total: </span>
                            <span className="fw-bold">${estimatedTotal}</span>
                        </div>
                    </div>
                    {/* HEREEEEE */}

                    <div className="modal-footer border-secondary">
                        <button className="btn btn-outline-dark" onClick={onHide}>Cancel</button>
                        <button
                            className={`btn ${orderType === 'BUY' ? 'btn-success' : 'btn-danger'}`}
                            onClick={() => handleTrade(onHide)}
                            disabled={!orderSize || orderSize <= 0}
                        >
                            Place {orderType} Order
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default TradeModal;