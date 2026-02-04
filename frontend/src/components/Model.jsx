import React, { useState } from 'react';


function Model() {
    const [picks, setPicks] = useState([]);
    const [loadingPicks, setLoadingPicks] = useState(false);

    const [singleTicker, setSingleTicker] = useState("");
    const [prediction, setPrediction] = useState(null);
    const [loadingPred, setLoadingPred] = useState(false);

    const [statusMsg, setStatusMsg] = useState("");

    const fetchNextDayPicks = async () => {
        setLoadingPicks(true);
        setStatusMsg("Fetching Next Day Picks...");
        try {
            const response = await fetch('http://127.0.0.1:8000/model/kelly-picks?version=A');
            if (!response.ok) throw new Error("Failed to fetch picks");
            const data = await response.json();
            setPicks(data);
            setStatusMsg("Loaded Next Day Picks.");
        } catch (e) {
            console.error(e);
            setStatusMsg("Error fetching picks.");
        } finally {
            setLoadingPicks(false);
        }
    };

    const fetchCurrentPicks = async () => {
        setLoadingPicks(true);
        setStatusMsg("Fetching Current Picks (Fastest Kelly)...");
        try {
            const response = await fetch('http://127.0.0.1:8000/model/current-picks');
            if (!response.ok) throw new Error("Failed to fetch current picks");
            const data = await response.json();
            setPicks(data);
            setStatusMsg("Loaded Current Picks.");
        } catch (e) {
            console.error(e);
            setStatusMsg("Error fetching current picks.");
        } finally {
            setLoadingPicks(false);
        }
    };

    const handlePredict = async () => {
        if (!singleTicker) return;
        setLoadingPred(true);
        setPrediction(null);
        try {
            const response = await fetch(`http://127.0.0.1:8000/model/predict/${singleTicker}`);
            if (!response.ok) throw new Error("Prediction failed");
            const data = await response.json();
            setPrediction(data);
        } catch (e) {
            console.error(e);
            setStatusMsg(`Error predicting for ${singleTicker}`);
        } finally {
            setLoadingPred(false);
        }
    };



    return (
        <div className="container mt-4 text-white">
            <h2 className="mb-4">Model Strategy & Analysis</h2>

            {statusMsg && <div className="alert alert-info">{statusMsg}</div>}

            <div className="d-flex gap-3 mb-4">
                <button className="btn btn-secondary" onClick={fetchCurrentPicks} disabled={loadingPicks}>
                    {loadingPicks ? "Loading..." : "Current Picks"}
                </button>
            </div>

            {/* Results Table */}
            {picks.length > 0 && (
                <div className="mb-5">
                    <h4>Allocation Recommendations</h4>
                    <p className="text-muted small">
                        Total Allocation: {(picks.reduce((sum, item) => sum + item.allocation, 0) * 100).toFixed(2)}%
                    </p>
                    <table className="table table-dark table-striped">
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Allocation</th>
                                <th>Proj Return</th>
                            </tr>
                        </thead>
                        <tbody>
                            {picks.map((pick) => (
                                <tr key={pick.ticker}>
                                    <td>{pick.ticker}</td>
                                    <td>{(pick.allocation * 100).toFixed(2)}%</td>
                                    <td>{(pick.projected_return * 100).toFixed(4)}%</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            <hr className="bg-secondary" />


            <div className="row mb-5 align-items-end">
                <div className="col-md-4">
                    <label className="form-label">Predict Single Ticker</label>
                    <div className="input-group">
                        <input
                            type="text"
                            className="form-control"
                            placeholder="Enter Ticker (e.g., AAPL)"
                            value={singleTicker}
                            onChange={(e) => setSingleTicker(e.target.value.toUpperCase())}
                        />
                        <button className="btn" onClick={handlePredict} disabled={loadingPred}>
                            {loadingPred ? "..." : "Predict"}
                        </button>
                    </div>
                </div>
                <div className="col-md-8">
                    {prediction && (
                        <div className="p-3 border border-secondary rounded bg-dark">
                            <strong>{prediction.ticker}</strong>:
                            Predicted Value: <span className="text-info">{prediction.prediction.toFixed(4)}</span> |
                            Last Close: <span className="text-warning">{prediction.last_close.toFixed(2)}</span>
                        </div>
                    )}
                </div>
            </div>

        </div>
    );
}

export default Model;